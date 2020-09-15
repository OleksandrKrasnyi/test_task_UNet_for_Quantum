import unet_model_core

train_gen = DataGen(train_ids, train_path,
                    image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path,
                    image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids) // batch_size
valid_steps = len(valid_ids) // batch_size

model.fit(train_gen,
          validation_data=valid_gen,
          steps_per_epoch=train_steps,
          validation_steps=valid_steps,
          epochs=epochs)

# Save model weights
model.save_weights("UNetWeigths.h5")
