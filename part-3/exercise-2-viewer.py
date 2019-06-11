import helper

images, labels = next(iter(trainloader))

img = images[0].view(1,784)

with torch.no_grad():
    logits = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = F.softmax(logits, dim=1)
helper.view_classify(img.view(1,28,28),ps)