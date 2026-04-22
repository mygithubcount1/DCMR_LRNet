import logging
import sys
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


logger = logging.getLogger(__name__)
def train_one_epoch_acc(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    total_num = len(data_loader.dataset)

    correct_num = torch.zeros(1).to(device)
    mean_acc = torch.zeros(1).to(device) 

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        original_images, vein_images, labels = data

        original_images = original_images.to(device)
        vein_images = vein_images.to(device)
        labels = labels.to(device)


        final_pred,pred1,pred2,pred3,pred4 = model(original_images, vein_images)


        pred_result = torch.max(final_pred, dim=1)[1]

        correct_num += torch.eq(pred_result, labels).sum().item() 

        loss1 = loss_function(pred1, labels)
        loss2 = loss_function(pred2, labels)
        loss3 = loss_function(pred3, labels)
        loss4 = loss_function(pred4, labels)
  
        loss = loss1 + loss2+loss3+loss4

        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  

        data_loader.desc = "[epoch {}] train loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            logger.warning(f'WARNING: non-finite loss, ending training. Loss: {loss}')
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    mean_acc = correct_num / total_num

    return mean_loss.item(), mean_acc.item()

@torch.no_grad()
def evaluate_acc(model, data_loader, device, epoch):
    model.eval()


    total_num = len(data_loader.dataset)
    total_loss = 0
    correct_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():  
        for step, data in enumerate(data_loader):

            (original_center_imgs, vein_center_imgs, original_top_imgs, vein_top_imgs,
             original_bottom_imgs, vein_bottom_imgs, labels) = data

  
            original_center_imgs = original_center_imgs.to(device)
            vein_center_imgs = vein_center_imgs.to(device)
            original_top_imgs = original_top_imgs.to(device)
            vein_top_imgs = vein_top_imgs.to(device)
            original_bottom_imgs = original_bottom_imgs.to(device)
            vein_bottom_imgs = vein_bottom_imgs.to(device)
            labels = labels.to(device)


            original_imgs_batch = torch.cat([
                original_center_imgs,
                original_top_imgs,
                original_bottom_imgs
            ], dim=0)

            vein_imgs_batch = torch.cat([
                vein_center_imgs,
                vein_top_imgs,
                vein_bottom_imgs
            ], dim=0)


            final_pred_batch, pred1, pred2, pred3, pred4 = model(original_imgs_batch, vein_imgs_batch)

            batch_size = original_center_imgs.shape[0]  
            final_center_pred = final_pred_batch[:batch_size]        
            final_top_pred = final_pred_batch[batch_size:2*batch_size] 
            final_bottom_pred = final_pred_batch[2*batch_size:]        
  

            final_pred = final_center_pred + final_top_pred + final_bottom_pred

            loss = torch.nn.CrossEntropyLoss()(final_pred, labels)
            total_loss += loss.item()


            pred = torch.max(final_pred, dim=1)[1]
            correct_num += torch.eq(pred, labels).sum().item()


    mean_loss = total_loss / len(data_loader)
    mean_acc = correct_num / total_num

    return mean_loss, mean_acc