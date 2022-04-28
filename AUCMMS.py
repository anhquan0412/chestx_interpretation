import torch 
import torch.nn.functional as F

class AUCM_MultiLabel_Sigmoid(torch.nn.Module):
    """
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, margin=1.0, imratio=[0.1], num_classes=10, device=None):
        super(AUCM_MultiLabel_Sigmoid, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.p = torch.FloatTensor(imratio).to(self.device)
        self.num_classes = num_classes
        assert len(imratio)==num_classes, 'Length of imratio needs to be same as num_classes!'
        self.a = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
        self.b = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_classes, dtype=torch.float32, device="cuda", requires_grad=True).to(self.device)

    @property
    def get_a(self):
        return self.a.mean()
    @property
    def get_b(self):
        return self.b.mean()
    @property
    def get_alpha(self):
        return self.alpha.mean()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)

        total_loss = 0
        for idx in range(self.num_classes):
            y_pred_i = y_pred[:, idx].reshape(-1, 1)
            y_true_i = y_true[:, idx].reshape(-1, 1)
            loss = (1-self.p[idx])*torch.mean((y_pred_i - self.a[idx])**2*(1==y_true_i).float()) + \
                        self.p[idx]*torch.mean((y_pred_i - self.b[idx])**2*(0==y_true_i).float())   + \
                        2*self.alpha[idx]*(self.p[idx]*(1-self.p[idx]) + \
                        torch.mean((self.p[idx]*y_pred_i*(0==y_true_i).float() - (1-self.p[idx])*y_pred_i*(1==y_true_i).float())) )- \
                        self.p[idx]*(1-self.p[idx])*self.alpha[idx]**2
            total_loss += loss
        return total_loss