import numpy as np


def get_X_mean_and_std(annotated_scenes):

    # Note: annotated scene shapes are (width, height, band)

    X_sizes = [X.size for X, *y in annotated_scenes]
    X_means = [X.mean(axis=(0, 1)) for X, *y in annotated_scenes]
    X_vars = [X.var(axis=(0, 1)) for X, *y in annotated_scenes]

    # Note: this produces the same result as
    #  np.hstack((X.flatten() for X, Y in annotated_scenes)).mean()
    # but uses less memory
    X_mean = np.average(X_means, weights=X_sizes, axis=0)
    X_var = np.average(X_vars, weights=X_sizes, axis=0)
    X_std = np.sqrt(X_var)

    X_mean = X_mean.reshape((1, 1, X_mean.size))
    X_std = X_std.reshape((1, 1, X_std.size))

    return X_mean, X_std


def get_X_normalized(X, X_mean, X_std):

    X_normalized = (X - X_mean) / X_std
    return X_normalized.astype(np.float32)


def normalize_scenes(annotated_scenes, X_mean, X_std):

    # Note: use training mean and std when normalizing validation and test scenes (avoid leakage)!

    for index, (X, *y) in enumerate(annotated_scenes):

        # Note: this modifies annotated_scenes in place
        annotated_scenes[index][0] = get_X_normalized(X, X_mean, X_std)
