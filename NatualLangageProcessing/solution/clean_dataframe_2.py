def clean_df_column(df, column_name, clean_column_name):
    nb_line = df.shape[0]
    df[clean_column_name] = [" ".join(apply_all_transformation(x)) for x in tqdm(df[column_name].values)]
