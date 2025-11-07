from matplotlib import pyplot as plt
import einops


def plot_batch_images(data, length=5, order="BT-CHW"):
    """
    Plot images in a batch.
    Args:
        data (torch.Tensor): Batch of images.
        length (int): Number of images
        order (str): Order of the dimensions in the data tensor.
    """
    if order == "BT-CHW":
        first_data = data[:length, :].detach().cpu().numpy()
        first_data = einops.rearrange(first_data, "l c h w -> l h w c")
    else:
        raise ValueError(f"Order {order} not supported.")

    for i in range(length):
        plt.figure(figsize=(10, 10))
        plt.imshow(first_data[i, :])

    plt.show()


def plot_action_sequence(observation, actions, obs_order="BT-CHW"):
    """
    Plot action sequence.
    Args:
        observation (torch.Tensor): Batch of observations.
        actions (torch.Tensor): Batch of actions.
        obs_order (str): Order of the dimensions in the observation tensor.
    """

    if obs_order == "BT-CHW":
        first_data = observation[0, :].detach().cpu().numpy()
        first_data = einops.rearrange(first_data, "1 c h w -> h w c")
    else:
        raise ValueError(f"Order {obs_order} not supported.")

    plt.figure(figsize=(10, 10))
    plt.imshow(first_data)

    first_actions = actions[0, :].detach().cpu().numpy()
    plt.figure(figsize=(10, 10))

    # Plot the actions with fading color
    alpha = 1
    for i in range(len(first_actions)):
        print(f"action: {first_actions[i]}")
        plt.scatter(first_actions[i][0], -first_actions[i][1], alpha=alpha, c="red")
        alpha = alpha * 0.9

    plt.show()
