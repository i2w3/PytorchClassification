import torch

@torch.no_grad()
def get_all_preds(model: torch.nn.Module,
                  loader: torch.utils.data.DataLoader) -> torch.tensor:
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch.cuda()

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds