import torch
from tqdm import tqdm

class WeightDecayEvalTrainer():
    def __init__(self) -> None:
        self.gradient_tracking = []
        pass

    def train(
        self, 
        model,
        gradient_monitoring_layer,
        criterion,
        optimizer,
        training_dataloader, 
        eval_dataloader,
        n_epochs=1, 
        print_loss_steps=10,
        grad_accum_steps=5,
        n_grad_eval_steps=100,
        max_training_steps=None,
        device="cuda:0",
    ):
        model.train()
        for epoch in range(n_epochs):
            with tqdm(training_dataloader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch, 0):
                    tepoch.set_description(f"Epoch {epoch}")

                    inputs, labels = data[0].to(device), data[1].to(device)

                    out = model(inputs)
                    loss = criterion(out, labels)
                    loss.backward()

                    if i % n_grad_eval_steps == 0:
                        self.gradient_tracking.append(torch.abs(gradient_monitoring_layer.weight.grad).mean().item())
                    
                    if i % grad_accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    
                    if i % print_loss_steps == 0:
                        tepoch.set_postfix(loss=loss.item())

                    if max_training_steps and len(training_dataloader)*epoch + i >= max_training_steps:
                        return

