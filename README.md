# Dc_GANS
Fake face generating model.
# Create data loader
CelebFaces Attributes Dataset is used for training network.
For working, a folder named train is required in the same directory with all the images.
Here we set the hyperparameters and created data loader.

# Generator and Discriminator
Both are trained at the same time. Discriminator is trained with both real and fake images while Generator is trained on only fake images.
By running this for a few epochs only, we got impressive results.
