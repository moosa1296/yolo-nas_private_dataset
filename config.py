root_dir ='/cluster/home/muhammmo/dataset_pigs'
train_images_dir = '/cluster/home/muhammmo/dataset_pigs/train'
train_labels_dir = '/cluster/home/muhammmo/dataset_pigs/train'
val_images_dir = '/cluster/home/muhammmo/dataset_pigs/val'
val_labels_dir = '/cluster/home/muhammmo/dataset_pigs/val'
test_images_dir = '/cluster/home/muhammmo/dataset_pigs/test'
test_labels_dir ='/cluster/home/muhammmo/dataset_pigs/test'
classes = ['pig']


dataset_params = {
    'data_dir': root_dir,
    'train_images_dir':train_images_dir,
    'train_labels_dir':train_labels_dir,
    'val_images_dir': val_images_dir,
    'val_labels_dir': val_labels_dir,
    'test_images_dir':test_images_dir,
    'test_labels_dir':test_labels_dir,
    'classes': classes
}