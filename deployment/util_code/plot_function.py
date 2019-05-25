"""By: Xiaochi (George) Li: github.com/XC-Li"""
import matplotlib.pyplot as plt


def summary_bar(experiment_table, title='Summary Plot'):
    """
    https://chrisalbon.com/python/data_visualization/matplotlib_grouped_bar_plot/
    :param experiment_table: Pandas dataframe
    :return: None
    """

    # Setting the positions and width for the bars
    pos = list(range(len(experiment_table['Combination'])))
    width = 0.2

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create a bar with test_neg data,
    # in position pos,
    plt.bar(pos,
            # using test_neg data,
            experiment_table['test_neg'],
            # of width
            width,
            # with alpha 0.5
            alpha=0.5,
            # with color
            color='#ff0000')

    plt.bar([p + width for p in pos],
            experiment_table['test_pos'],
            width,
            alpha=0.5,
            color='#ff5454')

    plt.bar([p + width * 2 for p in pos],
            experiment_table['train_neg'],
            width,
            alpha=0.5,
            color='#00ff00')

    plt.bar([p + width * 3 for p in pos],
            experiment_table['train_pos'],
            width,
            alpha=0.5,
            color='#63ff63')

    # Set the y axis label
    ax.set_ylabel('F1 Score')

    # Set the chart's title
    ax.set_title(title)

    # Set the position of the x ticks
    ax.set_xticks([p + 1.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(experiment_table['Combination'])
    for tick in ax.get_xticklabels():
        tick.set_rotation(70)

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])] )

    # Adding the legend and showing the plot
    plt.legend(['Test Negative', 'Test Positive', 'Train Negative', 'Train Positive'], loc='lower left')
    plt.grid()
    plt.show()
    