import os
import random
import logging

logger = logging.getLogger(__name__)

# NOTE: DO NOT MODIFY THE FOLLOWING PATHS
# ---------------------------------------
data_dir = os.environ.get("SM_CHANNEL_TRAIN", "../input/data")
# ---------------------------------------


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            boxes: (Optional) list. The bounding boxes of the words in the sequence.
            actual_bboxes: (Optional) list. The actual bounding boxes of the words in the sequence.
            file_name: (Optional) str. The name of the file.
            page_size: (Optional) list. The size of the page.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f, open(
        box_file_path, encoding="utf-8"
    ) as fb, open(image_file_path, encoding="utf-8") as fi:
        words = []
        boxes = []
        actual_bboxes = []
        file_name = None
        page_size = None
        labels = []
        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size,
                        )
                    )
                    guid_index += 1
                    words = []
                    boxes = []
                    actual_bboxes = []
                    file_name = None
                    page_size = None
                    labels = []
            else:
                splits = line.split("\t") # [word, label]
                bsplits = bline.split("\t") # [word, box]
                isplits = iline.split("\t") # [word, actual_bbox, page_size, file_name]
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                    box = bsplits[-1].replace("\n", "")
                    box = [int(b) for b in box.split()]
                    boxes.append(box)
                    actual_bbox = [int(b) for b in isplits[1].split()]
                    actual_bboxes.append(actual_bbox)
                    page_size = [int(i) for i in isplits[2].split()]
                    file_name = isplits[3].strip()
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words: # if file does not end with a newline
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                )
            )
    return examples



def write_examples_to_file(examples, mode):
    with open(f"{mode}.txt", "w", encoding="utf-8") as f, open(
        f"{mode}_box.txt", "w", encoding="utf-8"
    ) as fb, open(f"{mode}_image.txt", "w", encoding="utf-8") as fi:
        for example in examples:
            for word, label, box, actual_bbox in zip(
                example.words, example.labels, example.boxes, example.actual_bboxes
            ):
                f.write(f"{word}\t{label}\n")
                fb.write(f"{word}\t{' '.join([str(b) for b in box])}\n")
                fi.write(
                    f"{word}\t{' '.join([str(b) for b in actual_bbox])}\t{' '.join([str(i) for i in example.page_size])}\t{example.file_name}\n"
                )
            f.write("\n")
            fb.write("\n")
            fi.write("\n")


def main():
    train_examples = read_examples_from_file(data_dir, "train")

    random.Random(42)
    random.shuffle(train_examples)
    
    # split train and validation 8:2
    logger.info("Number of training examples: %d", len(train_examples))
    train_size = int(0.8 * len(train_examples))
    train_set = train_examples[:train_size]
    val_set = train_examples[train_size:]

    logger.info("Number of training set: %d", len(train_set))
    logger.info("Number of validation set: %d", len(val_set))

    write_examples_to_file(train_set, os.path.join(data_dir,"new_train"))
    write_examples_to_file(val_set, os.path.join(data_dir,"dev"))

    # rename the old files
    os.rename(os.path.join(data_dir, "train.txt"), os.path.join(data_dir, "old_train.txt"))
    os.rename(os.path.join(data_dir, "train_box.txt"), os.path.join(data_dir, "old_train_box.txt"))
    os.rename(os.path.join(data_dir, "train_image.txt"), os.path.join(data_dir, "old_train_image.txt"))

    os.rename(os.path.join(data_dir, "new_train.txt"), os.path.join(data_dir, "train.txt"))
    os.rename(os.path.join(data_dir, "new_train_box.txt"), os.path.join(data_dir, "train_box.txt"))
    os.rename(os.path.join(data_dir, "new_train_image.txt"), os.path.join(data_dir, "train_image.txt"))

    logger.info("Successfully split the training set into training and validation sets.")
    logger.info("The original training set is renamed to old_train.txt, old_train_box.txt, and old_train_image.txt.")

    
if __name__ == "__main__":
    main()