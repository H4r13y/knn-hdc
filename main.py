import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 1) Datensatz laden (digits)
# =====================================================================
def load_data():
    digits = load_digits()
    X_raw = digits.data      # shape (1797, 64)
    y_raw = digits.target    # 10 Klassen
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)
    return X_train, y_train, X_test, y_test

# =====================================================================
# 2) NN-Architekturen:
#    a) Klein:  (64 -> 64 -> 10)            (1 Hidden Layer)
#    b) Groß:   (64 -> 256 -> 128 -> 10)    (2 Hidden Layer)
# =====================================================================
class SmallMLP(nn.Module):
    """Kleines NN, 1 Hidden Layer (64)"""
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        return self.fc2(x)

    def extract_features(self, x):
        x = self.relu1(self.fc1(x))
        return x

class LargeMLP(nn.Module):
    """Größeres NN, 2 Hidden Layer (256 -> 128)"""
    def __init__(self, input_dim=64, hidden_dim1=256, hidden_dim2=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

    def extract_features(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x

def train_nn(model, loader, epochs=20, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

def eval_nn(model, X, y):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        out = model(X_t)
        preds = torch.argmax(out, dim=1).numpy()
    return np.mean(preds == y)

# =====================================================================
# 3) Einfaches HDC
# =====================================================================
class SimpleHDC:
    """
    Minimaler HDC-Ansatz mit +/-1-Vectoren.
    """
    def __init__(self, num_features=64, num_classes=10, D=2000, num_bins=30):
        self.num_features = num_features
        self.num_classes  = num_classes
        self.D = D
        self.num_bins = num_bins

        # Basis-HVs
        self.feature_hvs = (np.random.randint(2, size=(num_features, D)) * 2 - 1).astype(np.int8)
        # Level-HVs
        self.level_hvs   = (np.random.randint(2, size=(num_bins, D)) * 2 - 1).astype(np.int8)
        # Class-HVs
        self.class_hvs   = np.zeros((num_classes, D), dtype=np.int8)

        self.feature_mins = None
        self.feature_maxs = None

    def _encode_sample(self, x):
        accum = np.zeros(self.D, dtype=np.int32)
        for i in range(self.num_features):
            val = min(max(x[i], self.feature_mins[i]), self.feature_maxs[i])
            bin_idx = int((val - self.feature_mins[i]) /
                          (self.feature_maxs[i] - self.feature_mins[i] + 1e-9)* self.num_bins)
            bin_idx = min(bin_idx, self.num_bins - 1)
            bind_hv = self.feature_hvs[i] * self.level_hvs[bin_idx]
            accum += bind_hv
        # majority => +/-1
        hv_out = np.where(accum >= 0, 1, -1).astype(np.int8)
        return hv_out

    def fit(self, X_train, y_train):
        self.feature_mins = X_train.min(axis=0)
        self.feature_maxs = X_train.max(axis=0)
        accum_hvs = np.zeros((self.num_classes, self.D), dtype=np.int32)

        for x, label in zip(X_train, y_train):
            hv = self._encode_sample(x)
            accum_hvs[label] += hv

        self.class_hvs = np.where(accum_hvs >= 0, 1, -1).astype(np.int8)

    def predict(self, X):
        preds = []
        for x in X:
            qhv = self._encode_sample(x)
            sims = [np.sum(qhv * c_hv) for c_hv in self.class_hvs]
            best = np.argmax(sims)
            preds.append(best)
        return np.array(preds)

# =====================================================================
# 4) Hilfsfunktionen für das Experiment
# =====================================================================
def run_experiment(X_train, y_train, X_test, y_test,
                   nn_model_cls,  # z.B. SmallMLP oder LargeMLP
                   hdc_D,         # z.B. 512 oder 2000
                   epochs=20,
                   bins=30):
    """
    Trainiert + evaluiert:
     1) HDC-Only
     2) NN-Only (Model-Klasse = nn_model_cls)
     3) Hybrid
    Gibt ein Dictionary mit Ergebnissen zurück.
    """

    # 4.1 HDC-Only
    start = time.time()
    hdc = SimpleHDC(num_features=X_train.shape[1], num_classes=10, D=hdc_D, num_bins=bins)
    hdc.fit(X_train, y_train)
    t_hdc_train = time.time() - start

    # Train-Acc
    preds_train = hdc.predict(X_train)
    acc_train_hdc = np.mean(preds_train == y_train)

    start_infer = time.time()
    preds_test = hdc.predict(X_test)
    t_hdc_infer = time.time() - start_infer
    acc_test_hdc = np.mean(preds_test == y_test)

    # 4.2 NN-Only
    model = nn_model_cls()
    # DataLoader
    ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    start = time.time()
    train_nn(model, loader, epochs=epochs, lr=1e-3)
    t_nn_train = time.time() - start

    acc_train_nn = eval_nn(model, X_train, y_train)
    start_infer = time.time()
    acc_test_nn = eval_nn(model, X_test, y_test)
    t_nn_infer = time.time() - start_infer

    # 4.3 Hybrid
    # -> erst NN, dann HDC auf Hidden Features
    model_hyb = nn_model_cls()
    loader_hyb = DataLoader(TensorDataset(torch.FloatTensor(X_train),
                                          torch.LongTensor(y_train)),
                            batch_size=32, shuffle=True)
    start = time.time()
    train_nn(model_hyb, loader_hyb, epochs=epochs, lr=1e-3)
    # extrahiere features
    model_hyb.eval()
    with torch.no_grad():
        X_train_h = model_hyb.extract_features(torch.FloatTensor(X_train)).numpy()
        X_test_h  = model_hyb.extract_features(torch.FloatTensor(X_test)).numpy()
    # HDC
    hdc_hyb = SimpleHDC(num_features=X_train_h.shape[1], num_classes=10,
                        D=hdc_D, num_bins=bins)
    hdc_hyb.fit(X_train_h, y_train)
    t_hyb_train = time.time() - start

    # train-acc
    preds_hyb_train = hdc_hyb.predict(X_train_h)
    acc_train_hyb = np.mean(preds_hyb_train == y_train)

    start_infer = time.time()
    preds_hyb_test = hdc_hyb.predict(X_test_h)
    t_hyb_infer = time.time() - start_infer
    acc_test_hyb = np.mean(preds_hyb_test == y_test)

    results = {
        'HDC': {
            'train_time': t_hdc_train,
            'infer_time': t_hdc_infer,
            'train_acc': acc_train_hdc,
            'test_acc': acc_test_hdc
        },
        'NN': {
            'train_time': t_nn_train,
            'infer_time': t_nn_infer,
            'train_acc': acc_train_nn,
            'test_acc': acc_test_nn
        },
        'Hybrid': {
            'train_time': t_hyb_train,
            'infer_time': t_hyb_infer,
            'train_acc': acc_train_hyb,
            'test_acc': acc_test_hyb
        }
    }
    return results

# =====================================================================
# 5) Main: systematisches Durchprobieren
# =====================================================================
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    # Wir definieren einige Konfigurationen:
    # => NN-Typ: 'small' oder 'large'
    # => HDC-Dim: 512 oder 2000
    # => Epochen: 20

    configurations = [
        ('small', 512),
        ('small', 2000),
        ('large', 512),
        ('large', 2000)
    ]

    for (nn_size, hdc_dim) in configurations:
        if nn_size == 'small':
            nn_cls = SmallMLP
            label_str = "Small MLP"
        else:
            nn_cls = LargeMLP
            label_str = "Large MLP"

        print(f"\n===== CONFIG: NN={label_str}, HDC Dim={hdc_dim} =====")
        res = run_experiment(X_train, y_train, X_test, y_test,
                             nn_model_cls=nn_cls,
                             hdc_D=hdc_dim,
                             epochs=20,
                             bins=30)

        # Ausgabe
        hdc_res = res['HDC']
        nn_res  = res['NN']
        hyb_res = res['Hybrid']

        print(f"--- HDC-Only ---")
        print(f"Train Time: {hdc_res['train_time']:.3f}s, TrainAcc: {hdc_res['train_acc']:.3f}, "
              f"Infer Time: {hdc_res['infer_time']:.3f}s, TestAcc: {hdc_res['test_acc']:.3f}")

        print(f"--- NN-Only ---")
        print(f"Train Time: {nn_res['train_time']:.3f}s, TrainAcc: {nn_res['train_acc']:.3f}, "
              f"Infer Time: {nn_res['infer_time']:.3f}s, TestAcc: {nn_res['test_acc']:.3f}")

        print(f"--- Hybrid ---")
        print(f"Train Time: {hyb_res['train_time']:.3f}s, TrainAcc: {hyb_res['train_acc']:.3f}, "
              f"Infer Time: {hyb_res['infer_time']:.3f}s, TestAcc: {hyb_res['test_acc']:.3f}")

