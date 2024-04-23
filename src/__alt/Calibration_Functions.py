def plattScalingCal(val_dict, test_dict, test_dict_tensor):
    pred_dict = {}
    for group, data_val in val_dict.items():
        X_val = data_val['probabilities'][:]
        y_val = data_val['true_labels'][:]
        calibrated_clf.fit(X_val, y_val)
        X_test = test_dict[group]['probabilities'][:]
        pred_dict[group] = {}
        pred_dict[group]["probabilities"] = []
        cal_pred = calibrated_clf.predict_proba(X_test)
        for i in range(cal_pred.shape[0]):
            max_prob = torch.tensor(np.max(cal_pred[i,1]))
            pred_dict[group]["probabilities"].append(max_prob)
            pred_dict[group]["true_labels"] = test_dict_tensor[group]["true_labels"]
        #max_prob = []
        #for i in range(calibrated_prob.shape[0]):
            # Find the maximum probability in the second column (index 1)
           # max_prob = np.max(calibrated_prob[i, 1])
            # Append the maximum probability to the list
            #max_prob.append(max_prob)
        #pred_dict[group] = {}
        #pred_dict[group]["probabilities"] = max_prob
        #pred_dict[group]["true_labels"] = test_dict[group]['true_labels']
    return pred_dict

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