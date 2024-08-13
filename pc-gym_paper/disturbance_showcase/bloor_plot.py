def paper_plot(data):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 11
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['legend.fontsize'] = 11
 
    t = np.linspace(0, 25, 60)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
 
    policies = ['SAC', 'PPO', 'DDPG', 'oracle']
    cols = ['tab:red', 'tab:blue', 'tab:green', 'black']
    labels = ['SAC', 'PPO', 'DDPG', 'Oracle']
 
    # Create lines for the legend
    lines = []
    for i, policy in enumerate(policies):
        line, = axs[0].plot([], [], color=cols[i], label=labels[i])
        lines.append(line)
    ref_line, = axs[0].plot([], [], color='black', linestyle='--', label='Reference')
    lines.append(ref_line)
 
    # Create legend above the plots
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                ncol=5, frameon=False)
 
    y_labels = [r'$C_A$ [mol/m$^3$]', r'$T$ [K]', r'$T_c$ [K]']
   
    for idx, ax in enumerate(axs):
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
       
        if idx < 2:
            for i, policy in enumerate(policies):
                ax.plot(t, np.median(data[policy]['x'][idx,:,:], axis=1), color=cols[i])
                ax.fill_between(t, np.max(data[policy]['x'][idx,:,:], axis=1),
                                np.min(data[policy]['x'][idx,:,:], axis=1),
                                alpha=0.2, edgecolor='None', color=cols[i])
            if idx == 0:
                ax.step(t, data['SAC']['x'][2,:,0], color='black', linestyle='--')
        else:
            for i, policy in enumerate(policies):
                ax.step(t, np.median(data[policy]['u'][0,:,:], axis=1), color=cols[i], where='post')
                ax.fill_between(t, np.max(data[policy]['u'][0,:,:], axis=1),
                                np.min(data[policy]['u'][0,:,:], axis=1),
                                step="post", alpha=0.2, edgecolor='None', color=cols[i])
       
        ax.set_ylabel(y_labels[idx])
        ax.set_xlabel(r'Time (min)')
        ax.set_xlim(0, 25)
 
    # Adjust the plots to be square and the same size
    for ax in axs:
        ax.set_box_aspect(1)
       
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()