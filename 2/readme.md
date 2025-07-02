```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split #для разделения на тренировочные и тестовые выборки
from tqdm import tqdm
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
def make_regression_data(n=100, noise=0.2, source='random'):
    if source== 'random':
        X = torch.randn(n,1)
        w,b = -5, 10
        y = w*X + b + noise * torch.randn(n, 1)
        return X,y
    
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unk sources')
        
def mse(y_pred: torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true)**2).mean()
        
def log_epoch(epoch, avg_loss, **metrics):
    message = f'Epoch: {epoch}\t loss:{avg_loss}'
    for k, v in metrics.items():
        message += f'\t{k}: {v:.4f}'
    print(message)
    
def make_classification_data(n = 100):
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = torch.tensor(data['data'], dtype=torch.float32)
    y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
    return X, y

def make_multiclass_data(): #датасет с множеством классов
    from sklearn.datasets import load_wine
    data = load_wine()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = torch.tensor(data.target, dtype=torch.long) 
    return X, y

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred>0.5).float()
    return (y_pred_bin == y_true).float().mean().item()
```
# №1.1 LinearRegression
```python
import torch

class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
    
    def forward(self, x):
        return self.linear.forward(x)
    
```
```python
X, y = make_regression_data(10000)
EPOCHS = 100
dataset = CustomDataset(X, y)

# Решил разделить датасет на тренировочную и тестовую выборки, так как по определению early_stop ориентируется на ошибку 
# валидационной выборки
train_size = int(0.8 * len(dataset)) # train
val_size = int(0.2 * len(dataset)) # validate
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# dataloader = DataLoader(
#     dataset,
#     batch_size = 128,
#     shuffle = True,
# )

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

L1_coefficient = 0.01 # Добавили коэфициент влияния штрафа на Лассо
L2_coefficient = 0.01 # Добавили коэфициент влияния штрафа на Ridge 
Lasso = 0
Ridge = 0

early_stop = EPOCHS*0.4 # этот параметр нужен, чтобы иметь лимит при обучении, в случае увеличения эпох, качество ошибки не меняется или ниже
no_improvement = 0
best_val_loss = 1e-9
delta_number = 0.0001 # Добавил, чтобы модель учитывала не все улучшения, а только значимые. А то 1^(-10) улучшение как то несерьезно


lr = 0.5
epochs = 100
model = LinearRegressionTorch(1)

loss_fn = torch.nn.MSELoss() 

optimizer = torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(1, EPOCHS+1):
    total_loss = 0
    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model.forward(batch_x)
        
        
        if L1_coefficient != 0 or L2_coefficient != 0:    
            Lasso = 0
            Ridge = 0
            for name, param in model.named_parameters():
                if 'bias' not in name: # в формуле Лассо и Ridge мы суммируем только веса
                    # Ошибка по формуле sum(|wi|) * coef, а после param.abs ставим .sum, так как param - тензор
                    Lasso += param.abs().sum() 
                    # Ridge это то же Лассо, только веса беруться в квадрате
                    Ridge += (param ** 2).sum()
                    
        '''
        Сделал универсальный способ учета ошибки.
        Теперь это и LassoRegression в случае если будет указан только L1, и RidgeRegression если указывать только L2, 
        и ElasticNetRegression если указан и L1 и L2, и обычная линейная регрессия если ничего не указывать 
        
        Все по формуле ElasticNet = MSE + sum(|wi|) * coef_l1 + sum(wi^2) * coef_l2 
        '''
        loss = loss_fn.forward(y_pred, batch_y) + L1_coefficient * Lasso + L2_coefficient * Ridge
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader) #ошибка на обучающей выборке 

    model.eval() #переключаем модель в режим оценки 
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y) # Здесь штрафы не нужны, так как мы проверям качество модели в чистом виде
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    if avg_val_loss < best_val_loss-delta_number:
        no_improvement = 0
        best_val_loss = avg_val_loss
    else:
        no_improvement += 1

    if epoch % 10 == 0:
        log_epoch(epoch, avg_val_loss)
 
    if early_stop <= no_improvement:
        print(f"Early stopping на эпохе - {epoch}")
        break
    print(model.linear.weight.data, model.linear.bias.data)
```
# Вывод программы
```python
tensor([[-4.8982]]) tensor([9.9774])
tensor([[-4.9291]]) tensor([10.0438])
tensor([[-4.9449]]) tensor([10.0008])
tensor([[-4.9556]]) tensor([9.9876])
tensor([[-4.9473]]) tensor([9.9416])
tensor([[-4.9378]]) tensor([10.0318])
tensor([[-4.9459]]) tensor([9.9775])
tensor([[-4.9077]]) tensor([10.0229])
tensor([[-4.9153]]) tensor([9.9769])
Epoch: 10	 loss:0.03885782801080495
tensor([[-4.9569]]) tensor([10.0150])
tensor([[-4.9695]]) tensor([9.9823])
tensor([[-4.9543]]) tensor([10.0079])
tensor([[-4.9195]]) tensor([10.0269])
tensor([[-5.0076]]) tensor([9.9923])
tensor([[-4.9454]]) tensor([10.0006])
tensor([[-4.9591]]) tensor([10.0147])
tensor([[-4.9189]]) tensor([9.9636])
tensor([[-4.9344]]) tensor([9.9801])
tensor([[-4.9974]]) tensor([9.9804])
Epoch: 20	 loss:0.043882836354896426
tensor([[-4.9143]]) tensor([9.9965])
tensor([[-4.9712]]) tensor([9.9653])
tensor([[-4.9756]]) tensor([10.0291])
tensor([[-4.9594]]) tensor([10.0016])
tensor([[-4.9475]]) tensor([10.0185])
tensor([[-4.9409]]) tensor([9.9791])
tensor([[-4.9082]]) tensor([9.9697])
tensor([[-4.9100]]) tensor([9.9768])
tensor([[-4.9649]]) tensor([9.9932])
tensor([[-4.9474]]) tensor([10.0283])
Epoch: 30	 loss:0.04029263462871313
tensor([[-4.9473]]) tensor([10.0247])
tensor([[-4.9560]]) tensor([10.0051])
tensor([[-4.9445]]) tensor([10.0020])
tensor([[-4.9133]]) tensor([10.0141])
tensor([[-4.9682]]) tensor([10.0062])
tensor([[-4.9195]]) tensor([9.9846])
tensor([[-4.9657]]) tensor([10.0048])
tensor([[-4.9798]]) tensor([9.9763])
tensor([[-4.9614]]) tensor([10.0098])
tensor([[-4.9248]]) tensor([10.0592])
Epoch: 40	 loss:0.03698375774547458
Early stopping на эпохе - 40
```
# №1.2 LogisticRegression
```python
import torch

class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, in_features, n_classes=1):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, n_classes)
    
    def forward(self, x):
        return self.linear.forward(x)
```
```python
#Сделаю только для бинарной классификации, сложно как то для 1.5 часа решения домашки
def accuracy(y_pred, y_true): #вынес в отдельную функцию
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    return (y_pred == y_true).float().mean().item()

def precision(y_pred, y_true):
    y_pred = (y_pred.sigmoid() > 0.5).float()#Применяем sigmoid, чтобы получить вероятности от 0 до 1
    #это случаи (tp), когда модель сказала да (y_pred == 1) и на самом деле - да (y_true == 1) 
    tp = (y_pred * y_true).sum().item()
    #это случаи (fp), когда модель сказала да (y_pred == 1), а на самом деле - нет (y_true == 0).
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_pred, y_true):
    y_pred = (y_pred.sigmoid() > 0.5).float()
    tp = (y_pred * y_true).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    if p + r == 0:
        return 0.0
    return 2*p*r / (p + r) #изначально по формуле 2/(1/p + 1/r), но я раскрыл скобки просто

def roc_auc(y_pred, y_true):
    y_scores = y_pred.sigmoid().detach().numpy()#выход модели до sigmoid
    y_true = y_true.numpy()#реальные метки [0, 1]
    
    #cортировка по уверенности модели
    #мы делаем список пар (score, true_label) и сортируем его по убыванию уверенности модели.
    y_scores = y_scores.reshape(-1)  #делаем 1d (на случай, если пришёл 2d)
    y_true = y_true.reshape(-1)
    scores = np.column_stack((y_scores, y_true))  # связываем предсказания и правду
    scores = scores[scores[:, 0].argsort()][::-1]  # сортировка по убыванию уверенности
    
    #Общее число положительных и отрицательных примеров
    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    tp = fp = 0
    tpr_list = [0]
    fpr_list = [0]

    for score in scores:
        if score[1] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    #Вычисление площади под кривой (AUC)
    auc = 0
    for i in range(1, len(tpr_list)):
        #Интегрирование методом трапеций
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


def print_confusion_matrix(y_pred, y_true):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float().numpy().flatten()# Преобразуем в numpy для удобства
    y_true = y_true.numpy().flatten()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Выводим confusion matrix
    print("Confusion Matrix:")
    print("               | Predicted Pos | Predicted Neg |")
    print("---------------|--------------|--------------|")
    print(f"Actual Pos     |     {tp:4d}    |     {fn:4d}    |")
    print(f"Actual Neg     |     {fp:4d}    |     {tn:4d}    |")
    print("\n")

    # Считаем метрики
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Выводим метрики
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
```
```python
X, y = make_classification_data()
EPOCHS = 100
dataset = CustomDataset(X, y)
dataloader = DataLoader(
    dataset,
    batch_size = 16,
    shuffle = True,
)
lr = 0.1
epochs = 100

n_classes = len(torch.unique(y))# количество разнообразных классов
print(n_classes,'классы')

if n_classes > 2:
    model = LogisticRegressionTorch(X.shape[1], n_classes)# выбираем модель с несколькими классами
    loss_fn = torch.nn.CrossEntropyLoss() # Взял функцию для высчитывания ошибки в случае, если берем многоклассовый датасет
    '''
    С многоклассовым датасетом лучше работает Adam оптимизатор, 
    чем с SGD (SGD тоже работает нормально, но может застрять в локальных минимумах)
    '''
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
else:
    model = LogisticRegressionTorch(X.shape[1])# выбираем модель с несколькими классами
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

# Разделение на train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
    
for epoch in range(1, EPOCHS+1):
    model.train()#включаем режим обучения
    total_loss = 0
    total_acc = 0
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    total_auc = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model.forward(batch_x)
        loss = loss_fn.forward(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if n_classes > 2:
            preds = y_pred.argmax(dim=1)#Для accuracy сравниваем argmax предсказаний и реальных меток
            total_acc += (preds == batch_y).float().mean().item()
        else:
            probs = torch.sigmoid(y_pred)
            preds = (probs > 0.5).float()
            total_acc += (preds == batch_y).float().mean().item()
            
    avg_loss = total_loss / len(dataloader)
    avg_acc = total_acc/ len(dataloader)
    
    model.eval()#включаем для обучения
    all_preds = []
    all_probs = []
    all_true = []
    
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred_logits = model(batch_x)
            probs = torch.sigmoid(y_pred_logits)
            preds = (probs > 0.5).float().flatten()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(batch_y)

    all_preds = torch.tensor(all_preds)
    all_probs = torch.tensor(all_probs)
    all_true = torch.tensor(all_true)

    acc = accuracy(all_probs, all_true)
    prec = precision(all_probs, all_true)
    rec = recall(all_probs, all_true)
    f1 = f1_score(all_probs, all_true)
    auc = roc_auc(all_probs, all_true)
    
    
    if epoch % 10 == 0:
        print_confusion_matrix(torch.tensor(all_probs), torch.tensor(all_true))
        log_epoch(epoch, avg_loss, accuracy=avg_acc)

print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | "
              f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
```
# Вывод программы
```python
2 классы
C:\Users\user\AppData\Local\Temp\ipykernel_12296\3883148379.py:90: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  print_confusion_matrix(torch.tensor(all_probs), torch.tensor(all_true))
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       72    |        0    |
Actual Neg     |       39    |        3    |


Accuracy : 0.6579
Precision: 0.6486
Recall   : 1.0000
F1 Score : 0.7869
Epoch: 10	 loss:1983.7006427447002	accuracy: 0.6260
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       48    |       24    |
Actual Neg     |        2    |       40    |


Accuracy : 0.7719
Precision: 0.9600
Recall   : 0.6667
F1 Score : 0.7869
Epoch: 20	 loss:1264.7099523544312	accuracy: 0.6632
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       50    |       22    |
Actual Neg     |        1    |       41    |


Accuracy : 0.7982
Precision: 0.9804
Recall   : 0.6944
F1 Score : 0.8130
Epoch: 30	 loss:1749.1208685768975	accuracy: 0.6505
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       45    |       27    |
Actual Neg     |        1    |       41    |


Accuracy : 0.7544
Precision: 0.9783
Recall   : 0.6250
F1 Score : 0.7627
Epoch: 40	 loss:1098.5559719668495	accuracy: 0.6778
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       46    |       26    |
Actual Neg     |        1    |       41    |


Accuracy : 0.7632
Precision: 0.9787
Recall   : 0.6389
F1 Score : 0.7731
Epoch: 50	 loss:990.1389558580187	accuracy: 0.6982
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       46    |       26    |
Actual Neg     |        1    |       41    |


Accuracy : 0.7632
Precision: 0.9787
Recall   : 0.6389
F1 Score : 0.7731
Epoch: 60	 loss:908.7001145680746	accuracy: 0.7039
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       70    |        2    |
Actual Neg     |        6    |       36    |


Accuracy : 0.9298
Precision: 0.9211
Recall   : 0.9722
F1 Score : 0.9459
Epoch: 70	 loss:1086.0767902798123	accuracy: 0.6818
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       70    |        2    |
Actual Neg     |        4    |       38    |


Accuracy : 0.9474
Precision: 0.9459
Recall   : 0.9722
F1 Score : 0.9589
Epoch: 80	 loss:808.2126558091906	accuracy: 0.6979
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       70    |        2    |
Actual Neg     |        3    |       39    |


Accuracy : 0.9561
Precision: 0.9589
Recall   : 0.9722
F1 Score : 0.9655
Epoch: 90	 loss:666.8935457865397	accuracy: 0.7026
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       70    |        2    |
Actual Neg     |        5    |       37    |


Accuracy : 0.9386
Precision: 0.9333
Recall   : 0.9722
F1 Score : 0.9524
Epoch: 100	 loss:619.0000157296746	accuracy: 0.7217
Epoch 100 | Loss: 0.3381 | Acc: 0.9386, Prec: 0.9333, Rec: 0.9722, F1: 0.9524, AUC: 0.9289
```




# 2.1 Кастомный Dataset класс 
```python
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
#без знания pandas и возможностей sklearn я не знаю как решить задачу
class CustomDataset(Dataset):
    def __init__(self, file_path, target_col, numerical_cols=None, categorical_cols=None, transform=True,
                sep = None):
        """
        :param file_path: путь к .csv файлу
        :param target_col: имя целевой переменной
        :param numerical_cols: список числовых признаков
        :param categorical_cols: список категориальных признаков
        :param transform: применять ли нормализацию и кодирование
        """
        self.data = pd.read_csv(file_path, sep=sep)
        
        self.target_col = target_col
        
        # Разделение признаков и целевой переменной
        self.features = self.data.drop(columns=[target_col])
        self.targets = self.data[target_col].values
        
        # Автоматическое определение типов признаков
        if numerical_cols is None and categorical_cols is None:
            numerical_cols = []
            categorical_cols = []
            for col in self.features.columns:
                if self.features[col].dtype == 'O' or len(self.features[col].unique()) < 20:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        # Преобразование данных
        self.transform = transform
        if transform:
            self._apply_transform()

    def _apply_transform(self):
        """
        Применяет нормализацию и one-hot кодирование
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ],
            remainder='passthrough'
        )

        X_transformed = preprocessor.fit_transform(self.features)
        # Если матрица разреженная — переводим в плотный формат
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)

        self.X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        self.y_tensor = torch.tensor(self.targets, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]
        
def mse(y_pred: torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
    return ((y_pred - y_true)**2).mean()
        
def log_epoch(epoch, avg_loss, **metrics):
    message = f'Epoch: {epoch}\t loss:{avg_loss}'
    for k, v in metrics.items():
        message += f'\t{k}: {v:.4f}'
    print(message)

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred>0.5).float()
    return (y_pred_bin == y_true).float().mean().item()
```
# 2.2 Эксперименты с различными датасетами
```python
import torch

class LinearRegressionTorch(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
    
    def forward(self, x):
        return self.linear.forward(x)
import torch

class LogisticRegressionTorch(torch.nn.Module):
    def __init__(self, in_features, n_classes=1):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, n_classes)
    
    def forward(self, x):
        return self.linear.forward(x)
```
```python
url_linear = 'https://raw.githubusercontent.com/thainazanfolin/min_sleep_efficiency/refs/heads/main/Sleep_Efficiency_table.csv'
import kagglehub

url_logict = kagglehub.dataset_download("yasserh/titanic-dataset") #Взял все датасеты из своих наработок
#выбираем колонку по которой будем предиктить
# print(pd.read_csv(url_linear, sep=';'))
reg_dataset = CustomDataset(url_linear, target_col='Sleep efficiency', sep=';')

clf_dataset = CustomDataset(url_logict+'\\Titanic-Dataset.csv', target_col='Survived')

train_size_reg = int(0.8 * len(reg_dataset))
val_size_reg = len(reg_dataset) - train_size_reg
train_reg, val_reg = random_split(reg_dataset, [train_size_reg, val_size_reg])

train_loader_reg = DataLoader(train_reg, batch_size=32, shuffle=True)
val_loader_reg = DataLoader(val_reg, batch_size=32)

# То же самое для классификации
train_size_clf = int(0.8 * len(clf_dataset))
val_size_clf = len(clf_dataset) - train_size_clf
train_clf, val_clf = random_split(clf_dataset, [train_size_clf, val_size_clf])

train_loader_clf = DataLoader(train_clf, batch_size=32, shuffle=True)
val_loader_clf = DataLoader(val_clf, batch_size=32)
```
# Вывод программы
```python
Warning: Looks like you're using an outdated `kagglehub` version (installed: 0.3.10), please consider upgrading to the latest version (0.3.12).
C:\Users\user\AppData\Local\Temp\ipykernel_5672\886287871.py:20: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support sep=None with delim_whitespace=False; you can avoid this warning by specifying engine='python'.
  self.data = pd.read_csv(file_path, sep=sep)
```
# Обучение Линейной регрессии 
```python
EPOCHS = 100

# Решил разделить датасет на тренировочную и тестовую выборки, так как по определению early_stop ориентируется на ошибку 
# валидационной выборки
train_size = int(0.8 * len(reg_dataset)) # train
val_size = len(reg_dataset) - train_size # validate
train_dataset, val_dataset = random_split(reg_dataset, [train_size, val_size])

# dataloader = DataLoader(
#     dataset,
#     batch_size = 128,
#     shuffle = True,
# )

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

L1_coefficient = 0.01 # Добавили коэфициент влияния штрафа на Лассо
L2_coefficient = 0.01 # Добавили коэфициент влияния штрафа на Ridge 
Lasso = 0
Ridge = 0

early_stop = EPOCHS*0.4 # этот параметр нужен, чтобы иметь лимит при обучении, в случае увеличения эпох, качество ошибки не меняется или ниже
no_improvement = 0
best_val_loss = 1e-9
delta_number = 0.0001 # Добавил, чтобы модель учитывала не все улучшения, а только значимые. А то 1^(-10) улучшение как то несерьезно


lr = 0.5
epochs = 100
model = LinearRegressionTorch(reg_dataset.X_tensor.shape[1])

loss_fn = torch.nn.MSELoss() 

optimizer = torch.optim.SGD(model.parameters(),lr=lr)

for epoch in range(1, EPOCHS+1):
    total_loss = 0
    
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model.forward(batch_x)
        
        
        if L1_coefficient != 0 or L2_coefficient != 0:    
            Lasso = 0
            Ridge = 0
            for name, param in model.named_parameters():
                if 'bias' not in name: # в формуле Лассо и Ridge мы суммируем только веса
                    # Ошибка по формуле sum(|wi|) * coef, а после param.abs ставим .sum, так как param - тензор
                    Lasso += param.abs().sum() 
                    # Ridge это то же Лассо, только веса беруться в квадрате
                    Ridge += (param ** 2).sum()
                    
        '''
        Сделал универсальный способ учета ошибки.
        Теперь это и LassoRegression в случае если будет указан только L1, и RidgeRegression если указывать только L2, 
        и ElasticNetRegression если указан и L1 и L2, и обычная линейная регрессия если ничего не указывать 
        
        Все по формуле ElasticNet = MSE + sum(|wi|) * coef_l1 + sum(wi^2) * coef_l2 
        '''
        loss = loss_fn.forward(y_pred, batch_y) + L1_coefficient * Lasso + L2_coefficient * Ridge
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader) #ошибка на обучающей выборке 

    model.eval() #переключаем модель в режим оценки 
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y) # Здесь штрафы не нужны, так как мы проверям качество модели в чистом виде
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    if avg_val_loss < best_val_loss-delta_number:
        no_improvement = 0
        best_val_loss = avg_val_loss
    else:
        no_improvement += 1

    if epoch % 10 == 0:
        log_epoch(epoch, avg_val_loss)
 
    if early_stop <= no_improvement:
        print(f"Early stopping на эпохе - {epoch}")
        break
    # print(model.linear.weight.data, model.linear.bias.data)
```
# Вывод программы
```python
Epoch: 10	 loss:1.9074078928555633e+24
Epoch: 20	 loss:inf
Epoch: 30	 loss:inf
Epoch: 40	 loss:nan
Early stopping на эпохе - 40
```
# Добавление функций по высчитыванию параметров

```python
#Сделаю только для бинарной классификации, сложно как то для 1.5 часа решения домашки
def accuracy(y_pred, y_true): #вынес в отдельную функцию
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    return (y_pred == y_true).float().mean().item()

def precision(y_pred, y_true):
    y_pred = (y_pred.sigmoid() > 0.5).float()#Применяем sigmoid, чтобы получить вероятности от 0 до 1
    #это случаи (tp), когда модель сказала да (y_pred == 1) и на самом деле - да (y_true == 1) 
    tp = (y_pred * y_true).sum().item()
    #это случаи (fp), когда модель сказала да (y_pred == 1), а на самом деле - нет (y_true == 0).
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_pred, y_true):
    y_pred = (y_pred.sigmoid() > 0.5).float()
    tp = (y_pred * y_true).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    if p + r == 0:
        return 0.0
    return 2*p*r / (p + r) #изначально по формуле 2/(1/p + 1/r), но я раскрыл скобки просто

def roc_auc(y_pred, y_true):
    y_scores = y_pred.sigmoid().detach().numpy()#выход модели до sigmoid
    y_true = y_true.numpy()#реальные метки [0, 1]
    
    #cортировка по уверенности модели
    #мы делаем список пар (score, true_label) и сортируем его по убыванию уверенности модели.
    y_scores = y_scores.reshape(-1)  #делаем 1d (на случай, если пришёл 2d)
    y_true = y_true.reshape(-1)
    scores = np.column_stack((y_scores, y_true))  # связываем предсказания и правду
    scores = scores[scores[:, 0].argsort()][::-1]  # сортировка по убыванию уверенности
    
    #Общее число положительных и отрицательных примеров
    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos

    tp = fp = 0
    tpr_list = [0]
    fpr_list = [0]

    for score in scores:
        if score[1] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    #Вычисление площади под кривой (AUC)
    auc = 0
    for i in range(1, len(tpr_list)):
        #Интегрирование методом трапеций
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


def print_confusion_matrix(y_pred, y_true):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float().numpy().flatten()# Преобразуем в numpy для удобства
    y_true = y_true.numpy().flatten()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Выводим confusion matrix
    print("Confusion Matrix:")
    print("               | Predicted Pos | Predicted Neg |")
    print("---------------|--------------|--------------|")
    print(f"Actual Pos     |     {tp:4d}    |     {fn:4d}    |")
    print(f"Actual Neg     |     {fp:4d}    |     {tn:4d}    |")
    print("\n")

    # Считаем метрики
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Выводим метрики
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
```
# Обучение логистической регрессии
```python

EPOCHS = 100
lr = 0.1
epochs = 100

n_classes = 2# количество разнообразных классов
print(n_classes,'классы')

if n_classes > 2:
    model = LogisticRegressionTorch(X.shape[1], n_classes)# выбираем модель с несколькими классами
    loss_fn = torch.nn.CrossEntropyLoss() # Взял функцию для высчитывания ошибки в случае, если берем многоклассовый датасет
    '''
    С многоклассовым датасетом лучше работает Adam оптимизатор, 
    чем с SGD (SGD тоже работает нормально, но может застрять в локальных минимумах)
    '''
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
else:
    model = LogisticRegressionTorch(clf_dataset.X_tensor.shape[1])# выбираем модель с несколькими классами
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)

# Разделение на train/val
train_size = int(0.8 * len(clf_dataset))
val_size = len(clf_dataset) - train_size
train_dataset, val_dataset = random_split(clf_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
    
for epoch in range(1, EPOCHS+1):
    model.train()#включаем режим обучения
    total_loss = 0
    total_acc = 0
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    total_auc = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model.forward(batch_x)
        loss = loss_fn.forward(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if n_classes > 2:
            preds = y_pred.argmax(dim=1)#Для accuracy сравниваем argmax предсказаний и реальных меток
            total_acc += (preds == batch_y).float().mean().item()
        else:
            probs = torch.sigmoid(y_pred)
            preds = (probs > 0.5).float()
            total_acc += (preds == batch_y).float().mean().item()
            
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc/ len(train_loader)
    
    model.eval()#включаем для обучения
    all_preds = []
    all_probs = []
    all_true = []
    
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            y_pred_logits = model(batch_x)
            probs = torch.sigmoid(y_pred_logits)
            preds = (probs > 0.5).float().flatten()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_true.extend(batch_y)

    all_preds = torch.tensor(all_preds)
    all_probs = torch.tensor(all_probs)
    all_true = torch.tensor(all_true)

    acc = accuracy(all_probs, all_true)
    prec = precision(all_probs, all_true)
    rec = recall(all_probs, all_true)
    f1 = f1_score(all_probs, all_true)
    auc = roc_auc(all_probs, all_true)
    
    
    if epoch % 10 == 0:
        print_confusion_matrix(torch.tensor(all_probs), torch.tensor(all_true))
        log_epoch(epoch, avg_loss, accuracy=avg_acc)

print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | "
              f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
```
# Вывод программы
```python
2 классы
C:\Users\user\AppData\Local\Temp\ipykernel_5672\1616586711.py:83: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  print_confusion_matrix(torch.tensor(all_probs), torch.tensor(all_true))
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 10	 loss:0.40193198521931967	accuracy: 0.8278
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 20	 loss:0.3782078969809744	accuracy: 0.8403
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 30	 loss:0.355729270974795	accuracy: 0.8625
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 40	 loss:0.33801709844006433	accuracy: 0.8639
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 50	 loss:0.3283705747789807	accuracy: 0.8778
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 60	 loss:0.3114394168059031	accuracy: 0.8889
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 70	 loss:0.29436913728713987	accuracy: 0.8917
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 80	 loss:0.2846115903721915	accuracy: 0.8986
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 90	 loss:0.2684530324406094	accuracy: 0.9097
Confusion Matrix:
               | Predicted Pos | Predicted Neg |
---------------|--------------|--------------|
Actual Pos     |       78    |        0    |
Actual Neg     |      101    |        0    |


Accuracy : 0.4358
Precision: 0.4358
Recall   : 1.0000
F1 Score : 0.6070
Epoch: 100	 loss:0.25843607551521725	accuracy: 0.9139
Epoch 100 | Loss: nan | Acc: 0.4358, Prec: 0.4358, Rec: 1.0000, F1: 0.6070, AUC: 0.8247

```

# 3.1 Исследование гиперпараметров
```python
def train_regression_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=100, early_stop=20):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= early_stop:
                break
    
    return train_losses, val_losses
```
```python
def train_classification_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=100, early_stop=20):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    no_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            preds = (torch.sigmoid(y_pred) > 0.5).float()
            total_train_acc += (preds == batch_y).float().mean().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)
        
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                y_pred = model(batch_x)
                loss = loss_fn(y_pred, batch_y)
                total_val_loss += loss.item()
                preds = (torch.sigmoid(y_pred) > 0.5).float()
                total_val_acc += (preds == batch_y).float().mean().item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accs.append(avg_val_acc)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= early_stop:
                break
    
    return train_losses, val_losses, train_accs, val_accs
```
```python
# Визуализация результатов
def plot_results(df, x_col, y_cols, title, log_x=False):
    plt.figure(figsize=(12, 6))
    for y_col in y_cols:
        if log_x:
            plt.semilogx(df[x_col], df[y_col], marker='o', label=y_col)
        else:
            plt.plot(df[x_col], df[y_col], marker='o', label=y_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.legend()
    plt.grid(True)
    plt.show()
```
```python
def experiment_learning_rate_regression():
    lrs = [0.001, 0.01, 0.1, 0.5, 1.0]
    results = []
    
    for lr in lrs:
        model = LinearRegressionTorch(reg_dataset.X_tensor.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        
        train_losses, val_losses = train_regression_model(
            model, train_loader_reg, val_loader_reg, optimizer, loss_fn, epochs=100
        )
        
        results.append({
            'lr': lr,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'epochs': len(train_losses)
        })
    
    return pd.DataFrame(results)

# Эксперименты с batch size для регрессии
def experiment_batch_size_regression():
    batch_sizes = [8, 16, 32, 64, 128]
    results = []
    
    for batch_size in batch_sizes:
        train_loader = DataLoader(train_reg, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_reg, batch_size=batch_size)
        
        model = LinearRegressionTorch(reg_dataset.X_tensor.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.MSELoss()
        
        train_losses, val_losses = train_regression_model(
            model, train_loader, val_loader, optimizer, loss_fn, epochs=100
        )
        
        results.append({
            'batch_size': batch_size,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'epochs': len(train_losses)
        })
    
    return pd.DataFrame(results)

# Эксперименты с оптимизаторами для регрессии
def experiment_optimizers_regression():
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop
    }
    results = []
    
    for name, opt_class in optimizers.items():
        model = LinearRegressionTorch(reg_dataset.X_tensor.shape[1])
        optimizer = opt_class(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        train_losses, val_losses = train_regression_model(
            model, train_loader_reg, val_loader_reg, optimizer, loss_fn, epochs=100
        )
        
        results.append({
            'optimizer': name,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'epochs': len(train_losses)
        })
    
    return pd.DataFrame(results)

# Эксперименты с learning rate для классификации
def experiment_learning_rate_classification():
    lrs = [0.001, 0.01, 0.1, 0.5, 1.0]
    results = []
    
    for lr in lrs:
        model = LogisticRegressionTorch(clf_dataset.X_tensor.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        train_losses, val_losses, train_accs, val_accs = train_classification_model(
            model, train_loader_clf, val_loader_clf, optimizer, loss_fn, epochs=100
        )
        
        results.append({
            'lr': lr,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accs[-1],
            'best_val_loss': min(val_losses),
            'best_val_acc': max(val_accs),
            'epochs': len(train_losses)
        })
    
    return pd.DataFrame(results)

# Эксперименты с batch size для классификации
def experiment_batch_size_classification():
    batch_sizes = [8, 16, 32, 64, 128]
    results = []
    
    for batch_size in batch_sizes:
        train_loader = DataLoader(train_clf, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_clf, batch_size=batch_size)
        
        model = LogisticRegressionTorch(clf_dataset.X_tensor.shape[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        train_losses, val_losses, train_accs, val_accs = train_classification_model(
            model, train_loader, val_loader, optimizer, loss_fn, epochs=100
        )
        
        results.append({
            'batch_size': batch_size,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accs[-1],
            'best_val_loss': min(val_losses),
            'best_val_acc': max(val_accs),
            'epochs': len(train_losses)
        })
    
    return pd.DataFrame(results)

# Эксперименты с оптимизаторами для классификации
def experiment_optimizers_classification():
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop
    }
    results = []
    
    for name, opt_class in optimizers.items():
        model = LogisticRegressionTorch(clf_dataset.X_tensor.shape[1])
        optimizer = opt_class(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        train_losses, val_losses, train_accs, val_accs = train_classification_model(
            model, train_loader_clf, val_loader_clf, optimizer, loss_fn, epochs=100
        )
        
        results.append({
            'optimizer': name,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'final_val_acc': val_accs[-1],
            'best_val_loss': min(val_losses),
            'best_val_acc': max(val_accs),
            'epochs': len(train_losses)
        })
    
    return pd.DataFrame(results)

```
```python
# Запуск всех экспериментов и визуализация результатов
print("Running regression experiments...")
lr_reg_df = experiment_learning_rate_regression()
batch_reg_df = experiment_batch_size_regression()
opt_reg_df = experiment_optimizers_regression()

print("\nRunning classification experiments...")
lr_clf_df = experiment_learning_rate_classification()
batch_clf_df = experiment_batch_size_classification()
opt_clf_df = experiment_optimizers_classification()

# Визуализация результатов для регрессии
plot_results(lr_reg_df, 'lr', ['final_train_loss', 'final_val_loss'], 
            'Regression: Learning Rate vs Loss', log_x=True)
plot_results(batch_reg_df, 'batch_size', ['final_train_loss', 'final_val_loss'], 
            'Regression: Batch Size vs Loss')
plot_results(opt_reg_df, 'optimizer', ['final_train_loss', 'final_val_loss'], 
            'Regression: Optimizer vs Loss')

# Визуализация результатов для классификации
plot_results(lr_clf_df, 'lr', ['final_val_loss', 'final_val_acc'], 
            'Classification: Learning Rate vs Metrics', log_x=True)
plot_results(batch_clf_df, 'batch_size', ['final_val_loss', 'final_val_acc'], 
            'Classification: Batch Size vs Metrics')
plot_results(opt_clf_df, 'optimizer', ['final_val_loss', 'final_val_acc'], 
            'Classification: Optimizer vs Metrics')

# Вывод таблиц с результатами
print("\nRegression Learning Rate Results:")
print(lr_reg_df)

print("\nRegression Batch Size Results:")
print(batch_reg_df)

print("\nRegression Optimizer Results:")
print(opt_reg_df)

print("\nClassification Learning Rate Results:")
print(lr_clf_df)

print("\nClassification Batch Size Results:")
print(batch_clf_df)

print("\nClassification Optimizer Results:")
print(opt_clf_df)
```
# Вывод программы
```python
Running regression experiments...

Running classification experiments...
ДАЛЕЕ 6 ГРАФИКОВ (111,222,333,444,555,666) В ПАПКЕ PLOTS
Regression Learning Rate Results:
      lr  final_train_loss  final_val_loss  best_val_loss  epochs
0  0.001          0.005911        0.006742   6.741610e-03     100
1  0.010          0.002697        0.004074   4.073828e-03     100
2  0.100          0.001411        0.004843   4.502123e-03      41
3  0.500               NaN             NaN   1.264652e+09      21
4  1.000               NaN             NaN   2.335788e+18      21

Regression Batch Size Results:
   batch_size  final_train_loss  final_val_loss  best_val_loss  epochs
0           8          0.000076        0.003782       0.003636      45
1          16          0.001024        0.003497       0.003278      26
2          32          0.000726        0.004571       0.004430      61
3          64          0.002274        0.004165       0.003901      36
4         128          0.001833        0.003724       0.003703     100

Regression Optimizer Results:
  optimizer  final_train_loss  final_val_loss  best_val_loss  epochs
0       SGD          0.002747        0.003827       0.003801     100
1      Adam          0.000008        0.006681       0.005525      24
2   RMSprop          0.004131        0.011653       0.005683      50

Classification Learning Rate Results:
      lr  final_train_loss  final_val_loss  final_val_acc  best_val_loss  \
0  0.001          0.555908        0.559788       0.744518       0.559788   
1  0.010          0.432955        0.474268       0.773849       0.474268   
2  0.100          0.381062        0.474582       0.773849       0.471540   
3  0.500          0.246840        0.465506       0.785910       0.459768   
4  1.000          0.191144        0.468165       0.799890       0.454514   

   best_val_acc  epochs  
0      0.774123     100  
1      0.791393     100  
2      0.789748      47  
3      0.807018      46  
4      0.817434      38  

Classification Batch Size Results:
   batch_size  final_train_loss  final_val_loss  final_val_acc  best_val_loss  \
0           8          0.267111        0.475012       0.759058       0.474495   
1          16          0.379959        0.494730       0.751736       0.494198   
2          32          0.390283        0.474163       0.779057       0.469615   
3          64          0.414745        0.484346       0.769199       0.477952   
4         128          0.402639        0.485922       0.767080       0.484484   

   best_val_acc  epochs  
0      0.780797      51  
1      0.769097      24  
2      0.791393      35  
3      0.782271      42  
4      0.782782     100  

Classification Optimizer Results:
  optimizer  final_train_loss  final_val_loss  final_val_acc  best_val_loss  \
0       SGD          0.425862        0.475719       0.773849       0.475719   
1      Adam          0.069169        0.459654       0.815515       0.450208   
2   RMSprop          0.063393        0.471894       0.794682       0.453915   

   best_val_acc  epochs  
0      0.791393     100  
1      0.827851      36  
2      0.829496      28  
```

# Интерпретация результатов
Learning rate:

Слишком маленький LR приведет к медленному обучению

Слишком большой LR может вызвать расходимость

Оптимальное значение обычно между 0.001 и 0.1

Batch size:

Маленькие батчи могут привести к более шумным градиентам

Большие батчи требуют больше памяти

Оптимальный размер обычно между 16 и 64

Оптимизаторы:

SGD прост, но может требовать тонкой настройки LR

Adam более адаптивен, но может хуже обобщаться

RMSprop - компромисс между SGD и Adam


# Комментарии
Я где-то по середине 2-го задания стал намного меньше коментариев писать, заранее извините, у меня вся работ + оформление ReadME.md заняло очень много времени. 
Некоторые задания я не представляю выполнение без сторонних библиотек, таких как pandas, matplotlib, sklearn(а точнее отдельных разделов для one-hot, разделение датасета на тестовы и обучающие
для определенных условий без внедрения сторонних решений банально еще сильнее бы затянуло написание кода). Старался максимально подробно писать, возможно не стоило. 







Может быть вы это не читаете, я бы не читал :) , но
как интересный факт: я успел во время выполнения ДЗ успел прослушать полностью 2 фильма Властелина колец и почти досмотрел 3-ий (До момента, когда Арагорн приходит в пещеры Дунхарроу для договора с армией мертвецов)
