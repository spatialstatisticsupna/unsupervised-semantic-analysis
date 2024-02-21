
## CREATE A TIME SERIES OF GRIDS OF EMBEDDED TILES

X = np.zeros((n_samples, n_elems, z_dim))

tile_dir_grid = os.path.join(data_dir, "tiles")
if not os.path.exists(tile_dir_grid):
    os.makedirs(tile_dir_grid)
    
cont_t = 0
for filename in sorted(os.listdir(data_dir)):
    # Load image
    filename
    if filename.endswith(img_ext):
        img = load_img(os.path.join(data_dir, filename), val_type='uint8', bands_only=True)
        img = np.pad(img, pad_width=[(tile_radius, tile_radius),
                                            (tile_radius, tile_radius), (0, 0)], mode='reflect')
        cont_s = 0
        # For each tile in the grid of the given image
        for i in range(0, img.shape[0] // tile_size):
            for j in range(0, img.shape[1] // tile_size):
                start_r = i*tile_size
                end_r = (i+1)*tile_size
                start_c = j*tile_size
                end_c = (j+1)*tile_size
                # Save tile according to the time and number
                save_as = os.path.join(tile_dir_grid, '{sample}tile_T{t}.npy'.format(sample=cont_s, t=cont_t))
                np.save(save_as, img[start_r:end_r,start_c:end_c,:])
                # Create embedded vector
                tile = np.moveaxis(img[start_r:end_r,start_c:end_c,:], -1, 0)
                tile= np.expand_dims(tile, axis=0)
                tile = tile / 255
                 # Embed tile
                z = torch.from_numpy(tile).float()
                z = Variable(z)
                z = tilenet.encode(z)
                z = z.data.numpy()
                X[cont_s, cont_t, :] = z
                # Increase the count of samples within the image
                cont_s = cont_s + 1
        # Increase the time
        cont_t = cont_t + 1