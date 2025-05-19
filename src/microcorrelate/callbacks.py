"""
Registration callbacks taken from https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python
"""

import matplotlib.pyplot as plt

from IPython.display import clear_output


# Callback we associate with the StartEvent, sets up our new data.
def metric_start_plot():
    global metric_values, multires_iterations
    global current_iteration_number

    metric_values = []
    multires_iterations = []
    current_iteration_number = -1


# Callback we associate with the EndEvent, do cleanup of data and figure.
def metric_end_plot():
    global metric_values, multires_iterations
    global current_iteration_number

    del metric_values
    del multires_iterations
    del current_iteration_number
    # Close figure, we don't want to get a duplicate of the plot latter on
    plt.close()


# Callback we associate with the IterationEvent, update our data and display
# new figure.
def metric_plot_values(registration_method):
    global metric_values, multires_iterations
    global current_iteration_number

    # Some optimizers report an iteration event for function evaluations and not
    # a complete iteration, we only want to update every iteration.
    if registration_method.GetOptimizerIteration() == current_iteration_number:
        return

    current_iteration_number = registration_method.GetOptimizerIteration()
    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot
    # current data.
    clear_output(wait=True)
    # Plot the similarity metric values.
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.show()
