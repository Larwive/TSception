from cross_validation import *
from prepare_data_DEAP import *
import argparse
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ######## Data ########
    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='/home/dingyi/data/deap/')
    parser.add_argument('--start-subject', type=int, default=0)
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V'])
    parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1, 2, 512))
    parser.add_argument('--data-format', type=str, default='raw')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2021)
    parser.add_argument('--max-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc{}.pth')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--save-model', type=bool, default=True)
    parser.add_argument('--load-model', type=bool, default=False)
    parser.add_argument('--contribution', action='store_true')
    parser.add_argument('--load-all', action='store_true')
    parser.add_argument('--folding', action='store_true')
    parser.add_argument('--folds', type=int, default=10)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='TSception')
    parser.add_argument('--T', type=int, default=15)
    parser.add_argument('--graph-type', type=str, default='TS', choices=['TS', 'O'],
                        help='TS for the channel order of TSception, O for the original channel order')
    parser.add_argument('--hidden', type=int, default=32)

    ######## Reproduce the result using the saved model ######
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()

    sub_to_run_gen = np.arange(args.start_subject, args.start_subject + args.subjects)

    sub_to_runs = [
        sub_to_run_gen,
        sub_to_run_gen,
        [0],
        [0]
    ]

    train_labels = ["Arousal", "Valence", "Arousal (all subjects while training)", "Valence (all subjects while training)"]

    xlabels = ["Subject number's models",
               "Subject number's models",
               "Channels used",
               "Channels used"]

    folds = [
        10,
        10,
        10,
        10
    ]

    model_pathss = [[
        './all/save_ori/',
        './all/save_200/',
        './all/save_noF3_F4/',
        './all/save_Fp1_Fp2_AF3_AF4/',
        './all/save_Fp1_Fp2/',
        './all/save_AF3_AF4/',
        './all/save_all_except_4/',
        './all/save_Fp1_Fp2_T7_T8/',
        './all/save_T7_T8/',
        './all/save_Fp1_T7/',
        './all/save_Fp2_T8/',
        './all/save_O1_O2/'
    ], [
        './all/save_ori/',
        './all/save_all_except_4/',
        './all/save_Fp1_Fp2/',
        './all/save_AF3_AF4/',
        './all/save_T7_T8/',
    ], [
        #'./save_all/',
        './save_Fp1_Fp2_all/',
        './save_AF3_AF4_all/',
        './save_T7_T8_all/',
        './save_AF3_AF4_T7_T8_all/',
        './save_Fp1_Fp2_AF3_AF4_all/',
        './save_Fp1_Fp2_T7_T8_all/'
    ], [
        #'./save_all/',
        './save_Fp1_Fp2_all/',
        './save_AF3_AF4_all/',
        './save_T7_T8_all/',
        './save_AF3_AF4_T7_T8_all/',
        './save_Fp1_Fp2_AF3_AF4_all/',
        './save_Fp1_Fp2_T7_T8_all/'
    ]]

    input_shapess = [[
        (1, 28, 512),
        (1, 28, 512),
        (1, 26, 512),
        (1, 4, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 24, 512),
        (1, 4, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 2, 512),
    ], [
        (1, 28, 512),
        (1, 24, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 2, 512),
    ], [
        #(1, 28, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 4, 512),
        (1, 4, 512),
        (1, 4, 512)
    ], [
        #(1, 28, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 2, 512),
        (1, 4, 512),
        (1, 4, 512),
        (1, 4, 512)
    ]]

    labelss = [[
        "Original (500 epochs)",
        "All 28 channels (200 epochs)",
        "No F3, F4 (200 epochs)",
        "Fp1, Fp2, AF3, AF4 (200 epochs)",
        "Fp1, Fp2 (200 epochs)",
        "AF3, AF4 (200 epochs)",
        "All except Fp1, Fp2, AF3, AF4 (200 epochs)",
        "Fp1, Fp2, T7, T8 (200 epochs)",
        "T7, T8 (200 epochs)",
        "Fp1, T7 (200 epochs)",
        "Fp2, T8 (200 epochs)",
        "O1, O2 (200 epochs)",
    ], [
        "Original (500 epochs)",
        "All except Fp1, Fp2, AF3, AF4 (200 epochs)",
        "Fp1, Fp2 (200 epochs)",
        "AF3, AF4 (200 epochs)",
        "T7, T8 (200 epochs)",
    ], [
        #"All 28 channels (200 epochs)",
        "Fp1, Fp2 (200 epochs)",
        "AF3, AF4 (200 epochs)",
        "T7, T8 (200 epochs)",
        "AF3, AF4, T7, T8 (200 epochs)",
        "Fp1, Fp2, AF3, AF4 (200 epochs)",
        "Fp1, Fp2, T7, T8 (200 epochs)"
    ], [
        #"All 28 channels (200 epochs)",
        "Fp1, Fp2 (200 epochs)",
        "AF3, AF4 (200 epochs)",
        "T7, T8 (200 epochs)",
        "AF3, AF4, T7, T8 (200 epochs)",
        "Fp1, Fp2, AF3, AF4 (200 epochs)",
        "Fp1, Fp2, T7, T8 (200 epochs)"
    ]]

    channels_lists = [[
        ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
         'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
         'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'AF3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
        'Fp2', 'AF4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'AF3',
        'Fp2', 'AF4'],
        ['Fp1', 'Fp2'],
        ['AF3', 'AF4'],
        ['F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
        'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'T7',
        'Fp2', 'T8'],
        ['T7', 'T8'],
        ['Fp1', 'T7'],
        ['Fp2', 'T8'],
        ['O1', 'O2']
    ], [
        ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
         'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
        'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'Fp2'],
        ['AF3', 'AF4'],
        ['T7', 'T8'],
    ], [
        #['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
        # 'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'Fp2'],
        ['AF3', 'AF4'],
        ['T7', 'T8'],
        ['AF3', 'T7', 'AF4', 'T8'],
        ['Fp1', 'AF3', 'Fp2', 'AF4'],
        ['Fp1', 'T7', 'Fp2', 'T8']
    ], [
        #['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
        # 'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'],
        ['Fp1', 'Fp2'],
        ['AF3', 'AF4'],
        ['T7', 'T8'],
        ['AF3', 'T7', 'AF4', 'T8'],
        ['Fp1', 'AF3', 'Fp2', 'AF4'],
        ['Fp1', 'T7', 'Fp2', 'T8']
    ]]

    fign = 0

    plots = [2, 3]
    plotting = -1

    data_test, label_test = None, None
    for model_paths, input_shapes, labels, channels_lists, train_label, fold, sub_to_run, xlabel in zip(model_pathss,
                                                                                                        input_shapess,
                                                                                                        labelss,
                                                                                                        channels_lists,
                                                                                                        train_labels,
                                                                                                        folds,
                                                                                                        sub_to_runs,
                                                                                                        xlabels):
        plotting += 1
        if plotting not in plots:
            continue
        accuracies, stds = [], []
        args.label_type = train_label[0]
        for model_path, input_shape, label, channels in zip(model_paths, input_shapes, labels, channels_lists):
            accuracies_sub = []
            args.save_path = model_path
            args.input_shape = input_shape
            pd = PrepareData(args, channels)
            pd.run(sub_to_run, split=True, feature=False, expand=True, load_all=args.load_all)

            cv = CrossValidation(args)
            seed_all(args.random_seed)

            acc, std, accs, data_test, label_test = cv.compare(subject=sub_to_run, fold=fold, folding=args.folding,
                                                               data_test=data_test,
                                                               label_test=label_test,
                                                               load_all=(len(sub_to_run)==1))  # To do leave one trial out please set fold=40
            plt.figure(fign)
            plt.bar(np.arange(len(accs)), accs, label=label)
            for i in sub_to_run:
                accuracies_sub.append(np.array(accs[i * 10:(i + 1) * 10]).mean())
            plt.figure(fign + 1)
            plt.bar(np.arange(len(accuracies_sub)), accuracies_sub, label=label)
            accuracies.append(acc)
            stds.append(std)
            data_test, label_test = None, None

        plt.figure(fign)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.xlabel('Test number')
        plt.ylabel('Accuracy')
        plt.title(train_label)

        plt.figure(fign + 1)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.xlabel('Subject number\'s models')
        plt.ylabel('Accuracy')
        plt.title(train_label)

        print("{}\nmodel: acc/std".format(train_label))
        for model, acc, std in zip(model_paths, accuracies, stds):
            print("{}: {}/{}".format(model, acc, std))
        fign += 2

plt.show()

"""
Arousal
model: acc/std
./save_ori/: 0.5092473958333332/0.04143588425017974
./save_200/: 0.5057607421875/0.04354044933290499
./save_Fp1_Fp2/: 0.5218806966145833/0.04285639284138015
./save_AF3_AF4/: 0.5127032877604167/0.0468894105902508
./save_all_except_4/: 0.5261923828125/0.039551003748891894
./save_T7_T8/: 0.5202972005208333/0.04307661888267107
Valence
model: acc/std
./save_ori/: 0.5092473958333332/0.04143588425017974
./save_Fp1_Fp2/: 0.5057607421875/0.04354044933290499
./save_AF3_AF4/: 0.5218806966145833/0.04285639284138015
./save_all_except_4/: 0.5127032877604167/0.0468894105902508
./save_T7_T8/: 0.5261923828125/0.039551003748891894

"""
