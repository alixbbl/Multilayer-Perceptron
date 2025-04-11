import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# def plot_cost_report(dict_cost_report):
#     """
#       This functions will draw a cost report graph for each Hogwarts House, based on the training phase.
#     """
#     sns.set_theme(style="whitegrid")
#     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8)) # permet de decouper l'espace alloue au graphe en 4 portions egales
#     fig.tight_layout(pad=3.0)
    
#     houses = list(dict_cost_report.keys())
#     for i, house in enumerate(houses):
#         house_data = dict_cost_report[house] # chope la data de chaque maison
#         house_data_df = pd.DataFrame(house_data, columns=["Cost"])
#         house_data_df['Iterations'] = house_data_df.index
#         house_data_df['House'] = house
#         ax = axes[i // 2, i % 2] # permet de placer le graphe au bon endroit par rappor a son ordre d'affichage
#         sns.lineplot(data=house_data_df, x='Iterations', y='Cost', ax=ax, palette='Set2')
#         ax.set_title(house, fontsize=10)
#         ax.set_xlabel('Iterations', fontsize=8)
#         ax.set_ylabel('Log loss', fontsize=8)
#     fig.suptitle("Cost evolutions for each Hogwarts house", fontsize=16) # suptitle pour "super title" => global
#     plt.show()