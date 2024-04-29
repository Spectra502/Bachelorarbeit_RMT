import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import rand, randint
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

class Tscale(nn.Module):
    def __init__(self, dnn, lr, max_iter, lsf, valloader, testloader):
        super(Tscale, self).__init__()
        self.dnn = dnn
        self.t = nn.Parameter(torch.ones(1, device=device))

        self.crit = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.LBFGS([self.t], lr=lr, max_iter=max_iter, line_search_fn=lsf)
        self.valloader = valloader
        self.testloader = testloader
        self.device = device

    def fit(self):
        logits, labels = self.get_logits_labesl(self.valloader)

        def _closure():
            loss = self.crit(self.T_scaling(logits, self.t), labels)
            loss.backward()
            return loss

        self.optimizer.step(_closure)

    def T_scaling(self, logits, temperature):
        return torch.div(logits, temperature)

    def get_logits_labesl(self, loader):
        logits = []
        labels = []
        self.dnn.eval()
        #for idx, (x, y) in enumerate(loader):
        for batch_idx, (x, y, task) in enumerate(loader):
            if device != "cpu":
                #x, y = x.to(device=device), y.to(device=device)
                x, y = x.to(device=device), y

            with torch.no_grad():
                logits.append(self.dnn(x))
                labels.append(y)
        logits = torch.cat(logits).to(device)
        labels = torch.cat(labels).to(device)
        return logits, labels

    def test(self, n_bins: int):
        logits, labels = self.get_logits_labesl(self.testloader)
        with torch.no_grad():
            logits_scaled = self.T_scaling(logits, self.t)

        ece_builder = ECEevaluation(n_bins)
        return ece_builder.evaluate_from_logits(logits_scaled, labels)


def get_logits_labesl(loader, dnn):
        logits = []
        labels = []
        dnn.eval()
        #for idx, (x, y) in enumerate(loader):
        for batch_idx, (x, y, task) in enumerate(loader):
            if device != "cpu":
                #x, y = x.to(device=device), y.to(device=device)
                x, y = x.to(device=device), y

            with torch.no_grad():
                logits.append(dnn(x))
                labels.append(y)
        logits = torch.cat(logits).to(device)
        labels = torch.cat(labels).to(device)
        return logits, labels

def accuracy(predictions, targets):
    correct = sum([predictions[i] == targets[i] for i in range(len(predictions))])
    return correct/len(predictions)


class ECEevaluation(object):
    def __init__(self, n_bins: int):
        self.n_bins = n_bins

    def evaluate_from_logits(self, logits, labels):
        assert len(logits.shape) == 2
        assert len(labels.shape) == 1

        bin_borders = torch.linspace(0.0, 1.0, steps=self.n_bins+1)
        confs, preds = self._logits_to_conf_preds(logits)

        overall_acc = accuracy(preds, labels)

        bins = self._sort_to_bins(confs, preds, labels, bin_borders)

        n = len(labels)
        ece = 0.0
        for b in bins:
            if b["acc"] is not None:
                diff = abs(b["acc"]-b["conf"])
                ece += (len(b["confs"])/n)*diff
        return overall_acc.item(), ece, bins

    def _logits_to_conf_preds(self, logits):
        soft_probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(soft_probs, dim=1)
        return confs, preds

    def _sort_to_bins(self, confs, preds, labels, bin_borders):
        bins = [{
            "borders": (bin_borders[i].item(), bin_borders[i+1].item()),
        } for i in range(len(bin_borders)-1)]

        first = True
        for b in bins:
            if first:
                bool_lower = confs >= b["borders"][0]
                first = False
            else:
                bool_lower = confs > b["borders"][0]
            bool_upper = confs <= b["borders"][1]
            bool_in = bool_lower & bool_upper

            b["confs"] = confs[bool_in].tolist()
            b["preds"] = preds[bool_in].tolist()
            b["labels"] = labels[bool_in].tolist()

            if len(b["confs"]) > 0:
                b["acc"] = accuracy(b["preds"], b["labels"])
                b["conf"] = sum(b["confs"])/len(b["confs"])
            else:
                b["acc"] = None
                b["conf"] = None
        return bins
    

def run_ts_calib(dnn, hparams, test_loader, val_loader):

    vl_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # build datasets and dataloaders
    #testset = FSWDataset(paths=te_paths, labels=te_labels, transform=vl_transform)
    #valset = FSWDataset(paths=vl_paths, labels=vl_labels, transform=vl_transform)

    #val_dataloader = DataLoader(val_dataset_split, batch_size=2)
    #test_dataloader = DataLoader(test_dataset_split, batch_size=2)
    testloader = test_loader
    valloader = val_loader

    dnn = dnn
    #if DEVICE != "cpu":
        #dnn.to(DEVICE)

    ts = Tscale(dnn=dnn, lr=hparams["lr"], max_iter=hparams["max_iter"], lsf=hparams["lsf"],
                valloader=valloader, testloader=testloader)
    ts.fit()
    acc, ece, bins = ts.test(n_bins=15)

    print("-" * 150)
    print("test acc:\t", acc)
    print("test ece:\t", ece)

    logs = {
        "t parameter": ts.t.item(),
        "test acc": acc,
        "test ece": ece,
        "test bins": bins
    }

    #with open(f"../results/logs/{exp_id}_logs.json", "w") as file:
        #file.write(json.dumps(logs, indent=4))

    #with open(f"../results/hparams/{exp_id}_hp.json", "w") as file:
        #file.write(json.dumps(hparams, indent=4))


def run_ts_calibration_all_groups(dnn, hparams, loader):
    for i in range(0,7):
        print(f"Group {i+1}")
        ts = run_ts_calib(dnn, hparams, loader['test'][i], loader['validation'][i])
        print("\n\n")