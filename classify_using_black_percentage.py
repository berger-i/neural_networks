import math
from mnist import MNIST

mndata = MNIST(path='./MNIST', mode='rounded_binarized', return_type='numpy', gz=False)
images, labels = mndata.load_training()
#images = images[:10]
#labels = labels[:10]

images_testing, labels_testing = mndata.load_testing()

print('finished loading')

sums = {}
count = {}
for i in range(10):
    sums[i] = 0
    count[i] = 0

for image, label in zip(images, labels):
    sum_of_pixels = image.sum()
    sums[label] += sum_of_pixels
    count[label] += 1

means = {}
for label, sums_of_lable in sums.items():
    try:
        means[label] = sums_of_lable / count[label]
    except:
        pass


num_of_correct_answers = 0
num_of_wrong_answers = 0
for image, label in zip(images_testing, labels_testing):
    sum_of_pixels = image.sum()
    min_diff = math.inf
    ans = -1
    for means_label, means_mean in means.items():
        diff = abs(sum_of_pixels - means_mean)
        if diff < min_diff:
            min_diff = diff
            ans = means_label

    if ans == label:
        num_of_correct_answers += 1
    else:
        num_of_wrong_answers += 1

print('num_of_correct_answers', num_of_correct_answers)
print('num_of_wrong_answers', num_of_wrong_answers)
print('success rate:', num_of_correct_answers / (num_of_wrong_answers + num_of_wrong_answers))
