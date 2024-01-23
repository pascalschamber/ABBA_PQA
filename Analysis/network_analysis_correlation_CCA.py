import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA




#########################################################################################################
# contains fc+ext groups along with grouping info on first 4 columns
file_path = r"C:\Users\pasca\Box\Reijmers Lab\Pascal\Code\Image_Processing\quantification\network_analysis\data\2023_1109_fc-ext_zif_expression.xlsx"

#########################################################################################################
# Load the dataset to examine its structure and contents
data = pd.read_excel(file_path)

#########################################################################################################
# Split the data into the two groups
group_fc = data[data['group'] == 'FC']
group_fc_ext = data[data['group'] == 'FC+EXT']

# We'll perform t-tests on the cell densities across brain regions for the two groups
# Create a dictionary to hold the p-values for each brain region
p_values = {}

# Loop over all the brain regions (ignoring the first four columns)
for column in data.columns[4:]:
    # Perform the t-test between the two groups
    t_stat, p_val = ttest_ind(group_fc[column].dropna(), group_fc_ext[column].dropna(), equal_var=False)
    # Store the p-value in the dictionary
    p_values[column] = p_val

# Convert the dictionary to a DataFrame for easier analysis
p_values_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value'])

# Filter to include only significant p-values with a common threshold of alpha=0.05
significant_regions = p_values_df[p_values_df['p_value'] < 0.05].sort_values(by='p_value')

# Show the significant regions
significant_regions.head()


#########################################################################################################
# Filter the dataset to include only the cell density data
brain_regions_data = data.iloc[:, 4:]

# Standardize the data (very important for PCA)
scaler = StandardScaler()
brain_regions_scaled = scaler.fit_transform(brain_regions_data.dropna(axis=1))

# Fit PCA on the scaled data
pca = PCA(n_components=2)  # we choose 2 components for simplicity of visualization
principal_components = pca.fit_transform(brain_regions_scaled)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data = principal_components, columns = ['PC1', 'PC2'])

# Add the 'group' column to this dataframe for color-coding in visualization
pca_df['group'] = data['group']

# Show the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

explained_variance_ratio, pca_df.head()

#########################################################################################################
# Set the style of the visualization
sns.set(style="whitegrid")

# Create a scatter plot of the two principal components
plt.figure(figsize=(10,8))
sns.scatterplot(x="PC1", y="PC2", hue="group", data=pca_df, palette="viridis")

# Adding title and labels
plt.title('PCA of Brain Regions Cell Density')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Group', loc='best')
plt.show()

#########################################################################################################
# Calculate correlation matrices for each group
correlation_fc = group_fc.iloc[:, 4:].corr()
correlation_fc_ext = group_fc_ext.iloc[:, 4:].corr()

# We'll visualize the correlation matrix for the 'FC' group
plt.figure(figsize=(12,10))
sns.heatmap(correlation_fc, cmap='coolwarm')
plt.title('Correlation Matrix for FC Group')
plt.show()

# Let's also save the correlation matrix for the 'FC+EXT' group for later inspection
# We won't visualize it right now to avoid too many plots at once
# correlation_fc_ext.to_csv('/mnt/data/correlation_fc_ext_group.csv')

#########################################################################################################
# Prepare the features (X) and target (y)
X = brain_regions_data.dropna(axis=1)  # Drop columns with NaN to avoid errors during fitting
y = data['group'].loc[X.index]  # Ensure the target matches the features after dropna

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for easier analysis
feature_importances_df = pd.DataFrame({
    'Brain Region': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Show the top brain regions that are important for classification
feature_importances_df.head()

#########################################################################################################
# Initialize LDA
lda = LDA(n_components=1)  # We use 1 because we only have two groups

# Fit LDA to the training data
lda.fit(X_train, y_train)

# The coefficients of each feature in the LDA model
lda_coefs = lda.coef_[0]

# Create a DataFrame for easier analysis
lda_coefs_df = pd.DataFrame({
    'Brain Region': X_train.columns,
    'LDA Coefficient': lda_coefs
})

# Sort the DataFrame by the absolute values of the LDA coefficients
# Absolute value is used because both positive and negative coefficients are important for LDA
lda_coefs_df['Absolute LDA Coefficient'] = lda_coefs_df['LDA Coefficient'].abs()
lda_coefs_df = lda_coefs_df.sort_values(by='Absolute LDA Coefficient', ascending=False)

# Show the brain regions that contribute most to the separation between groups according to LDA
lda_coefs_df.head()

#########################################################################################################
# Since CCA requires the same number of samples in each group, we will create two datasets containing
# the same number of samples from each group.

# We will randomly sample from the larger group to match the number of samples in the smaller group.
group_fc_sampled = group_fc.sample(n=min(len(group_fc), len(group_fc_ext)), random_state=42)
group_fc_ext_sampled = group_fc_ext.sample(n=min(len(group_fc), len(group_fc_ext)), random_state=42)

# Prepare the data for CCA, ensuring there are no NaN values by dropping such columns
X_fc = group_fc_sampled.drop(columns=['animal_int', 'group', 'groupcohort', 'sex']).dropna(axis=1)
X_fc_ext = group_fc_ext_sampled.drop(columns=['animal_int', 'group', 'groupcohort', 'sex']).dropna(axis=1)

# Initialize CCA with 2 components as a starting point
cca = CCA(n_components=2)

# Fit CCA on the two groups
cca.fit(X_fc, X_fc_ext)

# Transform the data according to the CCA
X_c_fc, X_c_fc_ext = cca.transform(X_fc, X_fc_ext)

# Create a DataFrame to store CCA components and later visualize them
cca_components_df = pd.DataFrame({
    'CCA1_FC': X_c_fc[:, 0],
    'CCA2_FC': X_c_fc[:, 1],
    'CCA1_FC_EXT': X_c_fc_ext[:, 0],
    'CCA2_FC_EXT': X_c_fc_ext[:, 1]
})

# Show the first few rows of the CCA components
cca_components_df.head()

#########################################################################################################
# Visualize the relationships between the CCA components

plt.figure(figsize=(12, 12))
# Scatter plot for the first CCA components
# plt.subplot(1, 2, 1)
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.scatter(X_c_fc[:, 0], X_c_fc_ext[:, 0], color='blue', label='CCA1')
plt.scatter(X_c_fc[:, 1], X_c_fc_ext[:, 1], color='green', label='CCA2')
plt.title('Relationship between CCA1 Components')
plt.xlabel('CCA1_FC')
plt.ylabel('CCA1_FC_EXT')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 12))
plt.xlim(-15,15)
plt.ylim(-15,15)
# Scatter plot for the second CCA components
plt.scatter(X_c_fc[:, 0], X_c_fc[:, 1], color='red', label='FC')
plt.scatter(X_c_fc_ext[:, 0], X_c_fc_ext[:, 1], color='blue', label='EXT')
plt.title('PCA of first two components')
plt.xlabel('CCA1')
plt.ylabel('CCA2')
plt.legend()
plt.tight_layout()
plt.show()

#########################################################################################################
# Retrieve the canonical correlation coefficients
canonical_correlations = cca.score(X_fc, X_fc_ext)

# Retrieve the loadings for the brain regions
loadings_fc = cca.x_loadings_
loadings_fc_ext = cca.y_loadings_

# Create dataframes for the loadings of each group
loadings_fc_df = pd.DataFrame(loadings_fc, index=X_fc.columns, columns=['CCA1_FC', 'CCA2_FC'])
loadings_fc_ext_df = pd.DataFrame(loadings_fc_ext, index=X_fc_ext.columns, columns=['CCA1_FC_EXT', 'CCA2_FC_EXT'])

# Sort the loadings by the absolute values in descending order to identify the top contributing brain regions
top_loadings_fc_df = loadings_fc_df.abs().sort_values(by='CCA1_FC', ascending=False)
top_loadings_fc_ext_df = loadings_fc_ext_df.abs().sort_values(by='CCA1_FC_EXT', ascending=False)

canonical_correlations, top_loadings_fc_df.head(), top_loadings_fc_ext_df.head()

#########################################################################################################
# for each animal plot its value of the top 10 loadings from each group on x and y axis
top_n = 10
top_reg_fc = top_loadings_fc_df[:top_n].index.to_list()
top_reg_ext = top_loadings_fc_ext_df[:top_n].index.to_list()
fc_expression_fccca = group_fc[top_reg_fc].sum(axis=1).values
fc_expression_extcca = group_fc[top_reg_ext].sum(axis=1).values
ext_expression_fccca = group_fc_ext[top_reg_fc].sum(axis=1).values
ext_expression_extcca = group_fc_ext[top_reg_ext].sum(axis=1).values

plt.figure(figsize=(12, 12))
# plt.xlim(-15,15)
# plt.ylim(-15,15)
# Scatter plot for the second CCA components
plt.scatter(fc_expression_fccca, fc_expression_extcca, color='red', label='FC')
plt.scatter(ext_expression_fccca, ext_expression_extcca, color='blue', label='EXT')
plt.title('PCA of first two components')
plt.xlabel('CCA1')
plt.ylabel('CCA2')
plt.legend()
plt.tight_layout()
plt.show()
#########################################################################################################
# Since we need to overlay the corresponding region's loading from the other group on top of each bar, 
# we must first align the two sets of loadings by brain region. We will extract the corresponding loadings for the top regions of each group.
top_n = 20
# Mapping the top regions of one group to their corresponding loading in the other group
amygdala_regions = [el for el in top_loadings_fc_df.index.to_list() if 'amygdala' in el]
x_vals = amygdala_regions#top_loadings_fc_df.index
x_vals_ext = amygdala_regions
top_fc_loadings_in_fc_ext = loadings_fc_ext_df.loc[x_vals].fillna(0)#[:top_n] # top_loadings_fc_df.index
top_fc_ext_loadings_in_fc = loadings_fc_df.loc[x_vals_ext].fillna(0)#[:top_n] # top_loadings_fc_ext_df.index

# Plotting
plt.figure(figsize=(14, 10))

# Subplot for the 'FC' group loadings with 'FC+EXT' overlay
plt.subplot(2, 1, 1)
sns.barplot(y=x_vals, x='CCA1_FC', data=top_loadings_fc_df, color="red", label='FC Group')
sns.barplot(y=x_vals, x='CCA1_FC_EXT', data=top_fc_loadings_in_fc_ext, color="lightblue", label='FC+EXT Group Overlay')
plt.title('Top 10 Canonical Loadings for FC Group with FC+EXT Overlay')
plt.xlabel('CCA1 Loading Value')
plt.ylabel('Brain Region')
plt.legend()

# Subplot for the 'FC+EXT' group loadings with 'FC' overlay
plt.subplot(2, 1, 2)
sns.barplot(y=x_vals_ext, x='CCA1_FC_EXT', data=top_loadings_fc_ext_df, color="blue", label='FC+EXT Group')
sns.barplot(y=x_vals_ext, x='CCA1_FC', data=top_fc_ext_loadings_in_fc, color="pink", label='FC Group Overlay')
plt.title('Top 10 Canonical Loadings for FC+EXT Group with FC Overlay')
plt.xlabel('CCA1 Loading Value')
plt.ylabel('Brain Region')
plt.legend()

plt.tight_layout()
plt.show()

#########################################################################################################
# can I instead of above, find the loadings with the top differences between the two?


#########################################################################################################
# calculate the differences in the adjacency matricies between the two groups
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import numpy as np


def get_adjacency_matrix(df, significance_thresholds):
    print(f' og df shape: {df.shape}')
    # Calculate the correlation matrix
    correlation_matrix = df.corr(method='pearson')
    p_values = df.corr(method=lambda x, y: pearsonr(x, y)[1]).fillna(1)

    # Apply a significance filter and a false discovery rate of 5%
    rejected, p_values_fdr_bh, _, _ = multipletests(p_values.values.flatten(), alpha=significance_thresholds['multi_ttest_p'], method='fdr_bh')
    p_values_fdr_bh = p_values_fdr_bh.reshape(correlation_matrix.shape)

    # Binarize the correlation matrix to an adjacency matrix
    mask = (abs(correlation_matrix) > significance_thresholds['corr_thresh']) & (p_values_fdr_bh < significance_thresholds['multi_ttest_p'])
    adjacency_matrix = correlation_matrix.where(mask).notnull().astype('int')
    np.fill_diagonal(adjacency_matrix.values, 0)
    print(f'adjacency_matrix shape: {adjacency_matrix.shape}, sum edges: {adjacency_matrix.values.flatten().sum()}')
    
    return correlation_matrix, adjacency_matrix, p_values_fdr_bh


significance_thresholds = dict(
            multi_ttest_p = 0.05,
            corr_thresh = 0.80,
)

correlation_matrix_fc, adjacency_matrix_fc, p_values_fc = get_adjacency_matrix(group_fc.iloc[:, 4:], significance_thresholds)
correlation_matrix_ext, adjacency_matrix_ext, p_values_ext = get_adjacency_matrix(group_fc_ext.iloc[:, 4:], significance_thresholds)



plt.figure(figsize=(12,10))
sns.heatmap(adjacency_matrix_fc, cmap='coolwarm')
plt.title('Correlation Matrix for FC Group')
plt.show()

plt.figure(figsize=(12,10))
sns.heatmap(adjacency_matrix_ext, cmap='coolwarm')
plt.title('Correlation Matrix for adjacency_matrix_ext')
plt.show()

added_adjs = adjacency_matrix_fc + adjacency_matrix_ext
added_adjs_nan = np.where(added_adjs==0, np.nan, added_adjs)
added_adjs_nan2 = np.where(added_adjs_nan==2, np.nan, added_adjs_nan)
adjnan_fc = np.where(added_adjs_nan2==1, adjacency_matrix_fc, np.nan)
adjnan_ext = np.where(added_adjs_nan2==1, adjacency_matrix_ext, np.nan)
adjnan_comb = adjnan_fc - adjnan_ext # unique regions to each group
adjnan_comb2 = np.where(added_adjs_nan==2, 0, adjnan_comb) # shared correlations = 0, neither = nan

# generate a correlation matrix where fc +/- == 2/1, shared corrs == 0, and ext -/+ == -1,-2
adjnan_comb2_posneg_fc = np.where((adjnan_comb2==1) & (correlation_matrix_fc>0), 2, 0)
adjnan_comb2_posneg_fc = np.where((adjnan_comb2==1) & (correlation_matrix_fc<0), 1, adjnan_comb2_posneg_fc)
adjnan_comb2_posneg_ext = np.where((adjnan_comb2==-1) & (correlation_matrix_ext>0), -2, 0)
adjnan_comb2_posneg_ext = np.where((adjnan_comb2==-1) & (correlation_matrix_ext<0), -1, adjnan_comb2_posneg_ext)
adjnan_comb2_posneg = adjnan_comb2_posneg_fc + adjnan_comb2_posneg_ext
adjnan_comb2_posneg = np.where(adjnan_comb2_posneg==0, np.nan, adjnan_comb2_posneg)
adjnan_comb2_posneg = np.where(adjnan_comb2==0, 0, adjnan_comb2_posneg)

# do the same for the p values
p_values_adj = np.ones_like(p_values_fc)
p_values_adj = np.where(adjnan_comb2_posneg>0, p_values_fc, p_values_adj)
p_values_adj = np.where(adjnan_comb2_posneg<0, p_values_ext, p_values_adj)

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    'custom_colormap', 
    [(0.00, 'blue'), 
     (0.25, 'green'), 
     (0.50, 'grey'), 
     (0.75, 'yellow'), 
     (1.00, 'red')]
)

def heatmap_OpacityBySignificance(data, p_values, cmap):
    # Create the RGBA colormap based on the normalized p-values
    # p_values_normalized = 1 - p_values / p_values.max()
    
    # Adjust p-values slightly if there are zeros (since log(0) is undefined)
    p_values_adjusted = np.where(p_values == 0, np.min(p_values[p_values > 0]), p_values)

    # Use the negative log for the opacity to highlight small differences in p-values
    # The constant added to p_values_adj inside the log function is to avoid taking the log of zero
    log_p_values = -np.log10(p_values_adjusted)

    # Normalize the log p-values to the range [0, 1] for use as alpha values
    log_p_values_normalized = log_p_values / np.max(log_p_values)

    # log_p_values_normalized = 1-(p_values-p_values.min())/(0.05-p_values.min())

    # normalize the data to range (0,1)
    data = (data+2)/4

    rgba_data = np.zeros((data.shape[0], data.shape[1], 4))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # special cases
            if np.isnan(data[i, j]):
                base_color = [0,0,0,1]
                opacity = 1.0
            elif data[i, j] == 0.5:
                base_color = cmap(data[i, j])  # scale data to [0, 1] for the colormap # Get the base color from the custom colormap
                opacity = 1.0
            else:
                base_color = cmap(data[i, j])  # scale data to [0, 1] for the colormap # Get the base color from the custom colormap
                opacity = log_p_values_normalized[i, j]
            rgba_data[i, j, :3] = base_color[:3]  # RGB channels
            rgba_data[i, j, 3] = opacity  # Alpha channel
    return rgba_data, log_p_values_normalized

heatmap_rgb, p_values_adjusted = heatmap_OpacityBySignificance(adjnan_comb2_posneg, p_values_adj, cmap)




# Plotting the RGBA heatmap
fig, ax = plt.subplots(figsize=(12,12))
ax.imshow(heatmap_rgb, interpolation='nearest')
ax.set_facecolor('black')  # Set the background color to black
# Turn off the axis labels
ax.axis('off')
plt.savefig('heatmap_with_opacity.svg', dpi=300, bbox_inches='tight')
plt.show()



fig,ax = plt.subplots(figsize=(12,10))
sns.heatmap((adjnan_comb2_posneg+2)/4, cmap=cmap, ax=ax)
ax.set_facecolor('black')
plt.title('Correlation Matrix for adjnan_comb2_posneg')
plt.show()
