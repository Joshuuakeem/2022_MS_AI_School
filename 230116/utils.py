import torch
def train(num_epoch, model, train_loader, val_loader, criterion, optimizer,
          scheduler, device) :
    print("training.......!!!")
    total = 0
    best_loss = 9999
    for epoch in range(num_epoch) :
        for i , (imgs, labels) in enumerate(train_loader) :
            imgs, labels = imgs.to(device), labels.to(device)

            ouptut = model(imgs)

            loss = criterion(ouptut, labels)
            scheduler.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _ , argmax = torch.max(ouptut, 1)
            acc = (labels == argmax).float().mean()
            total += labels.size(0)

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, '
                      'Accuracy: {:.2f}% '.format(
                    epoch + 1, num_epoch, i + 1, len(train_loader),
                    loss.item(), acc.item() * 100))

        avrg_loss, val_acc = validation(model, val_loader, criterion, device)

        if avrg_loss < best_loss :
            print("best model save !!")
            best_loss = avrg_loss
            torch.save(model.state_dict(), "./best.pt")

    torch.save(model.state_dict(), "./last.pt")

def validation(model, val_loader, criterion, device) :
    print("val ....")
    model.eval()
    with torch.no_grad() :
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        for i, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)

    print('Validation Accuracy: {:.2f}%  Average Loss: {:.4f}'.format(
        correct / total * 100, avrg_loss))

    model.train()

    return avrg_loss, val_acc