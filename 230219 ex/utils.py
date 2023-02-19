import torch
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def visualize_aug(dataset, idx=0, samples=10, cols=5) :
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([
        t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))
    ])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12,6))

    for i in range(samples) :
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

def train(num_epoch, model, train_loader, val_loader, criterion,
          optimizer, scheduler, save_dir, device) :
    print("Strart training.........")
    running_loss = 0.0
    total = 0
    best_loss = 9999
    for epoch in range(num_epoch+1) :
        for i, (imgs, labels) in enumerate(train_loader) :
            img, label = imgs.to(device) , labels.to(device)

            output = model(img)
            loss = criterion(output, label)
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, argmax = torch.max(output, 1)
            acc = (label==argmax).float().mean()
            total += label.size(0)

            if (i + 1 ) % 10 == 0:
                print("Epoch [{}/{}] Step[{}/{}] Loss :{:.4f} Acc : {:.2f}%".format(
                    epoch + 1 , num_epoch, i+1, len(train_loader), loss.item(),
                    acc.item() * 100
                ))


        avrg_loss, val_acc = validation(epoch, model, val_loader, criterion,
                                        device)
        # if epoch % 10 == 0 :
        #     save_model(model, save_dir, file_naem=f"{epoch}.pt")
        if avrg_loss < best_loss :
            print(f"Best save at epoch >> {epoch}")
            print("save model in " , save_dir)
            best_loss = avrg_loss
            save_model(model, save_dir)

    save_model(model, save_dir, file_name="last_resnet.pt")