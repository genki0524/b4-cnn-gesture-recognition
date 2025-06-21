import time
import copy
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Tuple
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    学習処理の基本構造を定義する抽象基底クラス。
    継承先でtrainメソッドを必ず実装する必要があります。
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        dataset_sizes: Dict[str, int],
        device: torch.device,
        num_epochs: int
    ) -> None:
        """
        BaseTrainerの初期化。

        Args:
            model (nn.Module): 学習対象のモデル
            criterion (nn.Module): 損失関数
            optimizer (Optimizer): 最適化手法
            scheduler (_LRScheduler): 学習率スケジューラ
            dataloaders (dict): データローダ（train/test）
            dataset_sizes (dict): データセットサイズ
            device (torch.device): 使用デバイス
            num_epochs (int): 学習エポック数
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
        self.num_epochs = num_epochs

    @abstractmethod
    def train(self) -> nn.Module:
        """
        学習処理を実装する抽象メソッド。
        継承先で必ず実装してください。

        Returns:
            nn.Module: 学習済みモデル
        """
        pass

class Trainer(BaseTrainer):
    """
    モデルの学習処理を実装するクラス。
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        dataset_sizes: Dict[str, int],
        device: torch.device,
        num_epochs: int
    ) -> None:
        """
        Trainerの初期化。

        Args:
            model (nn.Module): 学習対象のモデル
            criterion (nn.Module): 損失関数
            optimizer (Optimizer): 最適化手法
            scheduler (_LRScheduler): 学習率スケジューラ
            dataloaders (dict): データローダ（train/test）
            dataset_sizes (dict): データセットサイズ
            device (torch.device): 使用デバイス
            num_epochs (int): 学習エポック数
        """
        super().__init__(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs)

    def train(self) -> nn.Module:
        """
        モデルの学習を実行します。

        Returns:
            nn.Module: 学習済みモデル
        """
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}\n{"-" * 10}')

            for phase in ['train', 'val']:
                running_loss, running_corrects = self.run_epoch(
                    model=self.model,
                    dataloaders=self.dataloaders,
                    device=self.device,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    phase=phase
                )

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 検証精度がベストならモデルを保存
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # ベストモデルの重みをロード
        self.model.load_state_dict(best_model_wts)
        return self.model

    @staticmethod
    def run_epoch(
        model: nn.Module,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        device: torch.device,
        optimizer: Optimizer,
        criterion: nn.Module,
        phase: str
    ) -> Tuple[float, int]:
        """
        1エポック分の学習または検証処理を行います。

        Args:
            model (nn.Module): モデル
            dataloaders (dict): データローダ
            device (torch.device): デバイス
            optimizer (Optimizer): 最適化手法
            criterion (nn.Module): 損失関数
            phase (str): 'train' または 'val'

        Returns:
            Tuple[float, int]: 合計損失、正解数
        """
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 順伝播
            with torch.set_grad_enabled(phase == 'train'):
                representation, outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, representation, labels,device)

                # 訓練の時だけ逆伝播＋オプティマイズを行います
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 損失を計算します
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        return running_loss, running_corrects

