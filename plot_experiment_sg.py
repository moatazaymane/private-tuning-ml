
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

exp =2

if exp == 1:
    rares = np.linspace(0.02, 0.1, 10)
    mean_diffs = np.load("results/_5_sg_sh_mean_diffs.npy")
    std_diffs = np.load("results/_5_sg_sh_std_diffs.npy")

    rs_mean_diffs = np.load("results/5_sg_rs_mean_diffs.npy")
    rs_std_diffs = np.load("results/5_sg_rs_std_diffs.npy")


    plt.figure(figsize=(8, 5))
    from matplotlib.ticker import MaxNLocator

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.set(style="whitegrid")
    print(mean_diffs)
    plt.plot(100*rares, mean_diffs, linestyle="-", color="#0073e6", label=r"$SH$")
    plt.fill_between(100*rares, np.clip(mean_diffs - std_diffs, 0, 1), mean_diffs + std_diffs, 
                    color="#0073e6", alpha=0.2)

    plt.plot(100*rares, rs_mean_diffs, linestyle="-", color="#B22222", label=r"$RS$")
    plt.fill_between(100*rares, np.clip(rs_mean_diffs - rs_std_diffs, 0, 1), rs_mean_diffs + rs_std_diffs, 
                    color="#B22222", alpha=0.2)

    plt.xlabel(r"$r$", fontsize=12)
    plt.ylabel(r"$\mathcal{R}_{\mathrm{simple}}$", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/_sh_regret_vs_rare.png", dpi=300)
    plt.show()


else:

    C_values = np.arange(3, 8, 1)
    mean_diffs = np.clip(np.load("results/_C_sg_sh_mean_diffs.npy"), 0, 1)
    std_diffs = np.clip(np.load("results/_C_sg_sh_std_diffs.npy"), 0, 1)

    rs_mean_diffs = np.clip(np.load("results/C_sg_rs_mean_diffs.npy"), 0, 1)
    rs_std_diffs = np.clip(np.load("results/C_sg_rs_std_diffs.npy"), 0, 1)

    plt.figure(figsize=(8, 5))

    from matplotlib.ticker import MaxNLocator

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    sns.set(style="whitegrid")
    print(mean_diffs)
    plt.plot(C_values.astype(int), mean_diffs, linestyle="-", color="#0073e6", label=r"$SH$")
    plt.fill_between(C_values.astype(int), np.clip(mean_diffs - std_diffs, 0, 1), mean_diffs + std_diffs, 
                    color="#0073e6", alpha=0.2)

    plt.plot(C_values.astype(int), rs_mean_diffs, linestyle="-", color="#B22222", label=r"$RS$")
    plt.fill_between(C_values.astype(int), np.clip(rs_mean_diffs - rs_std_diffs, 0, 1), rs_mean_diffs + rs_std_diffs, 
                    color="#B22222", alpha=0.2)

    plt.xlabel(r"$\epsilon$", fontsize=12)
    plt.ylabel(r"$\mathcal{R}_{\mathrm{simple}}$", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/_C_sh_regret_vs_rare.png", dpi=300)
    plt.show()

