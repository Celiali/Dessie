import pandas as pd
import glob, os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the file')
    parser.add_argument("--FLAG", action="store_true", default=False,help="FLAG True: PA ; False: PC")
    args = parser.parse_args()

    # Step 1: Use glob to get all CSV files
    path = args.path 
    # Step 1: Read all CSV files matching the naming pattern
    version_order = [9]
    csv_files = [os.path.join(path,f"Eval_PFERD_version_{t}_eval_results.csv") for t in version_order] # Adjust path pattern if necessary
    print(csv_files)

    # PA True: read PA; PA False: read PCK
    PA = args.FLAG 

    # Initialize an empty list to hold data from each CSV
    data_list = []

    # Step 2: Loop through each CSV file and load the data
    for file in csv_files:
        df = pd.read_csv(file)
        # Add the version to the DataFrame
        version = os.path.basename(file).split('_')[3]  # Extract version from filename (adjust if needed)
        df['version'] = version
        filtered_df = df[~df['dataset'].str.contains("20201128_ID_1_0005")]
        data_list.append(filtered_df)

    # Step 3: Combine all DataFrames into one
    combined_df = pd.concat(data_list, ignore_index=True)

    # Step 4: Filter the DataFrame to focus on 'mean' flag only
    if PA == True :
        mean_df = combined_df[combined_df['flag'] == 'mean'] 
    else:
        mean_df = combined_df[combined_df['flag'] == 'pck_mean']

    # Step 5: Group by 'version' and 'metric_name', and calculate the mean of 'metric_value'
    grouped_df = mean_df.groupby(['version', 'metric_name'])['metric_value'].mean().reset_index()

    # Step 6: Pivot the DataFrame to have 'metric_name' as columns
    pivot_df = grouped_df.pivot(index='version', columns='metric_name', values='metric_value').reset_index()

    # Step 7: Rename the columns for clarity
    pivot_df.columns = ['version'] + [f'average_{metric}' for metric in pivot_df.columns if metric != 'version']

    # Step 8: Define the custom order for the versions
    custom_order = ['9']

    # Step 9: Reorder the DataFrame based on the custom version order
    pivot_df['version'] = pd.Categorical(pivot_df['version'], categories=custom_order, ordered=True)
    pivot_df = pivot_df.sort_values('version').reset_index(drop=True)


    # Step 10: Define the custom order for the columns
    if PA == True:
        custom_columns = ['version'] + [
            'average_mode_mpjpe_rigid', 'average_mode_re_rigid'

        ]
    else:
        custom_columns = ['version'] + [
            'average_mode_rigid_pck10'
        ]


    # Step 11: Reorder the columns in the DataFrame
    pivot_df = pivot_df[custom_columns]

    # Step 12: Print the result with tab-separated values
    print(pivot_df.to_csv(sep='\t', index=False))