from sklearn.model_selection import train_test_split


def split_by_student(seq_df, test_size=0.2, val_size=0.1, seed=42):
    students = seq_df["student_id"].unique()

    train_ids, test_ids = train_test_split(
        students, test_size=test_size, random_state=seed
    )

    train_df = seq_df[seq_df["student_id"].isin(train_ids)].copy()
    test_df = seq_df[seq_df["student_id"].isin(test_ids)].copy()

    return train_df, test_df
