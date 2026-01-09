from baselines import get_baseline
from datasets import get_dataset, list_datasets


if __name__ == "__main__":

    dataset = get_dataset("Textures")
    print(dataset.class_names)

    # ind_dataset = get_dataset(args.ind_dataset)
    # ood_dataset = get_dataset(args.ood_dataset)