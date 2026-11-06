import os
import itertools
import copy

import laspy
import numpy as np
import pandas as pd
import CSF

# input dataset

datasets = {
    "bare_sand": [
        "dataset_ground_filtering/bare_sand1.laz",
        "dataset_ground_filtering/bare_sand2.las"],
    "sparse_vegetation": [
        "dataset_ground_filtering/sparse_vegetation1.las",
        "dataset_ground_filtering/sparse_vegetation2.las"],
    "dense_vegetation": [
        "dataset_ground_filtering/dense_vegetation1.las",
        "dataset_ground_filtering/dense_vegetation2.las"]}

output_folder = "dataset_ground_filtering/out/CSF_sensitivity_analysis"

os.makedirs(output_folder, exist_ok=True)

# csf parameters

cloth_resolutions = [0.5, 1.0, 2.0]
thresholds = [0.25, 0.5]
rigidness_values = [1, 2]

iterations = 500
time_step = 0.65
slope_smoothing = False

parameter_combinations = list(
    itertools.product(
        cloth_resolutions,
        thresholds,
        rigidness_values))

# Jaccard similarity function

def jaccard_similarity(class_a, class_b):

    intersection = np.sum(class_a & class_b)
    union = np.sum(class_a | class_b)

    if union == 0: # avoid 0 division error
        return 1.0

    return intersection / union


# Start analysis

all_results = []
all_sensitivity_results = []

for dataset_name, filepaths in datasets.items():

    print(f"\nPROCESSING DATASET: {dataset_name}")

    for filepath in filepaths:
        las = laspy.read(filepath)
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
        total_points = len(xyz)

        if total_points == 0:
            print("Empty point cloud. Skipping.")
            continue

        file_stem = os.path.splitext(os.path.basename(filepath))[0]


        ground_masks = []
        combination_records = []

        # run all parameter combinations

        for combination_id, (resolution, threshold, rigidness) in enumerate(parameter_combinations):
            csf = CSF.CSF()
            csf.params.cloth_resolution = resolution
            csf.params.class_threshold = threshold
            csf.params.rigidness = rigidness
            csf.params.time_step = time_step
            csf.params.interations = iterations
            csf.params.bSloopSmooth = slope_smoothing
            csf.setPointCloud(xyz)
            ground = CSF.VecInt()
            non_ground = CSF.VecInt()
            csf.do_filtering(ground, non_ground)
            ground_idx = np.fromiter(ground, dtype=np.int64)
            non_ground_idx = np.fromiter(non_ground, dtype=np.int64)

            # Check classification indices

            if (np.any(ground_idx < 0)
                or np.any(ground_idx >= total_points)
                or np.any(non_ground_idx < 0)
                or np.any(non_ground_idx >= total_points)):
                raise ValueError("CSF returned invalid point indices.")

            if np.intersect1d(ground_idx, non_ground_idx).size > 0:
                raise ValueError("Ground and non-ground indices overlap.")

            if len(ground_idx) + len(non_ground_idx) != total_points:
                raise ValueError("Some points were not classified.")

            # Create Boolean ground mask

            ground_mask = np.zeros(total_points, dtype=bool)
            ground_mask[ground_idx] = True
            ground_masks.append(ground_mask)

            ground_percentage = (len(ground_idx) / total_points * 100)

            combination_records.append({
                "dataset": dataset_name,
                "input_file": file_stem,
                "combination_id": combination_id,
                "resolution": resolution,
                "threshold": threshold,
                "rigidness": rigidness,
                "ground_points": len(ground_idx),
                "non_ground_points": len(non_ground_idx),
                "ground_percentage": ground_percentage})

        # Convert classifications to NumPy matrix

        ground_masks = np.asarray(ground_masks)
        number_combinations = len(ground_masks)

        # pairwise classification similarity

        similarity_matrix = np.zeros((number_combinations, number_combinations))

        for i in range(number_combinations):
            for j in range(number_combinations):
                similarity_matrix[i, j] = jaccard_similarity(
                    ground_masks[i],
                    ground_masks[j]
                )

        # Mean similarity of each combination against all other combinations

        mean_similarity = (
            similarity_matrix.sum(axis=1) - 1
        ) / (number_combinations - 1)

        for i in range(number_combinations):
            record = combination_records[i].copy()
            record["mean_jaccard_similarity"] = mean_similarity[i]
            all_results.append(record)

        # Parameter sensitivity analysis

        parameter_definitions = {
            "cloth_resolution": cloth_resolutions,
            "threshold": thresholds,
            "rigidness": rigidness_values}

        parameter_columns = {
            "cloth_resolution": 0,
            "threshold": 1,
            "rigidness": 2}

        for parameter_name, parameter_values in parameter_definitions.items():

            parameter_position = parameter_columns[parameter_name]

            other_positions = [
                p for p in range(3)
                if p != parameter_position]

            isolated_similarities = []

            # Compare combinations where ONLY the selected parameter changes.

            for other_values in itertools.product(
                *[parameter_definitions[name]
                for name in parameter_definitions
                if parameter_columns[name] in other_positions]):

                matching_indices = []

                for i, combination in enumerate(parameter_combinations):
                    if all(
                        combination[pos] == value
                        for pos, value in zip(other_positions, other_values)):
                        matching_indices.append(i)

                # Compare every pair of values for the selected parameter

                for i, j in itertools.combinations(matching_indices, 2):
                    similarity = jaccard_similarity(
                        ground_masks[i],
                        ground_masks[j])
                    isolated_similarities.append(similarity)

            mean_agreement = np.mean(isolated_similarities)
            sensitivity = 1 - mean_agreement

            all_sensitivity_results.append({
                "dataset": dataset_name,
                "input_file": file_stem,
                "parameter": parameter_name,
                "mean_jaccard_agreement": mean_agreement,
                "sensitivity_score": sensitivity})


# Export result of analysis to csv
pd.DataFrame(all_results).to_csv(os.path.join(output_folder,"combination_similarity.csv"),index=False)
pd.DataFrame(all_sensitivity_results).to_csv(os.path.join(output_folder,"parameter_sensitivity.csv"),index=False)

print(f"Results saved in: {output_folder}")