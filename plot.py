import numpy as np
import matplotlib.pyplot as plt


def plot_train_test_losses(single_agent:np.array, multi_agent:np.array,multi_agent_genetic,
                           title="Reward Per Episode",
                           x_label="Episode Steps", y_label="Reward",
                           min_max_bounds= True,
                           tight_x_lim = True, y_lim=None,
                           signle_legend = "Single Agent",
                           multi_legend = "Multi Agent",
                           genetic_legend = "Multi Agent + Genetic",
                           save_path=None)->None:

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = 12
    mean_single_rewards = np.nanmean(single_agent, axis=0)
    std_single_rewards = np.nanstd(single_agent, axis=0)
    mean_multi_rewards= np.nanmean(multi_agent, axis=0)
    std_multi_rewards = np.nanstd(multi_agent, axis=0)
    mean_genetic_rewards= np.nanmean(multi_agent_genetic, axis=0)
    std_genetic_rewards = np.nanstd(multi_agent_genetic, axis=0)
    
    if min_max_bounds:
        lower_single_rewards = np.nanmin(single_agent, axis=0)
        upper_single_rewards = np.nanmax(single_agent, axis=0)
        lower_multi_rewards = np.nanmin(multi_agent, axis=0)
        upper_multi_rewards= np.nanmax(multi_agent, axis=0)
        lower_genetic_rewards = np.nanmin(multi_agent_genetic, axis=0)
        upper_genetic_rewards= np.nanmax(multi_agent_genetic, axis=0)
        
        
    else:
        lower_single_rewards = mean_single_rewards - std_single_rewards
        upper_single_rewards = mean_single_rewards + std_single_rewards
        lower_multi_rewards = mean_multi_rewards - std_multi_rewards
        upper_multi_rewards = mean_multi_rewards + std_multi_rewards
        lower_genetic_rewards = mean_genetic_rewards - std_genetic_rewards
        upper_genetic_rewards = mean_genetic_rewards + std_genetic_rewards
        

    x_range = range(1, len(mean_single_rewards) + 1)
    
    plt.plot(x_range ,mean_single_rewards, color='#33a9a5', linewidth=2, label=signle_legend)
    plt.fill_between(x_range, lower_single_rewards, upper_single_rewards, alpha=0.2, color='#33a9a5', edgecolor='none')

    plt.plot(x_range ,mean_multi_rewards, color='#f27085', linewidth=2, label=multi_legend)
    plt.fill_between(x_range, lower_multi_rewards, upper_multi_rewards, alpha=0.2, color='#f27085', edgecolor='none')
    
    plt.plot(x_range ,mean_genetic_rewards, color='#f2c085', linewidth=2, label=genetic_legend)
    plt.fill_between(x_range, lower_genetic_rewards, upper_genetic_rewards, alpha=0.2, color='#f2c085', edgecolor='none')
    

    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_lim is not None:
        plt.ylim(y_lim)
    if tight_x_lim:
        plt.xlim(1, single_agent.shape[1])
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())
def convert2uint8(x):
    return (x * 255).astype(np.uint8)

def display_images(array1, array2, names, title, figsize = (10,5), savepath=None):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for ax, array, name in zip(axs, [array1, array2], names):
        ax.imshow(array)
        ax.set_title(name)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    fig.suptitle(title)
    # tighten the plot
    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    train_losses = np.random.random((10, 100)) * np.geomspace(100, 1, num=100, endpoint=True)  /100 
    test_losses = np.random.random((10, 100)) * np.geomspace(100, 1, num=100, endpoint=True)  /100 + np.linspace(.1, 0, num=100, endpoint=True) + 0.05

    plot_train_test_losses(train_losses, test_losses,y_lim=[0,1])
