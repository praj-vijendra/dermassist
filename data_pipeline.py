import os
import uuid
import json
import pandas as pd
import ast
import shutil


def list_files(directory):
    """Helper to list files in a directory (useful for debugging)."""
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    for file in files:
        print(file)


def initialize_df_with_metadata(csv_path):
    """Loads the given CSV into a pd.DataFrame."""
    df = pd.read_csv(csv_path, dtype={"case_id": str})
    df["case_id"] = df["case_id"].astype(str)
    return df


def augment_metadata_with_labels(df, csv_path):
    """Loads the given CSV into a pd.DataFrame and merges with the existing DataFrame."""
    labels_df = pd.read_csv(csv_path, dtype={"case_id": str})
    labels_df["case_id"] = labels_df["case_id"].astype(str)
    merged_df = pd.merge(df, labels_df, on="case_id")
    return merged_df


def process_dataframe(cases_and_labels_df):
    """
    Processes the input DataFrame to create a new DataFrame with consolidated information.
    """
    column_groups = [
        {
            "columns": [
                "image_1_path",
                "image_1_shot_type",
                "dermatologist_gradable_for_skin_condition_1",
                "dermatologist_gradable_for_fitzpatrick_skin_type_1",
                "dermatologist_fitzpatrick_skin_type_label_1",
            ],
            "modified_columns": [
                "image_path",
                "shot_type",
                "dermatologist_gradable_for_skin_condition",
                "dermatologist_gradable_for_fitzpatrick_skin_type",
                "dermatologist_fitzpatrick_skin_type_label",
            ],
        },
        {
            "columns": [
                "image_2_path",
                "image_2_shot_type",
                "dermatologist_gradable_for_skin_condition_2",
                "dermatologist_gradable_for_fitzpatrick_skin_type_2",
                "dermatologist_fitzpatrick_skin_type_label_2",
            ],
            "modified_columns": [
                "image_path",
                "shot_type",
                "dermatologist_gradable_for_skin_condition",
                "dermatologist_gradable_for_fitzpatrick_skin_type",
                "dermatologist_fitzpatrick_skin_type_label",
            ],
        },
        {
            "columns": [
                "image_3_path",
                "image_3_shot_type",
                "dermatologist_gradable_for_skin_condition_3",
                "dermatologist_gradable_for_fitzpatrick_skin_type_3",
                "dermatologist_fitzpatrick_skin_type_label_3",
            ],
            "modified_columns": [
                "image_path",
                "shot_type",
                "dermatologist_gradable_for_skin_condition",
                "dermatologist_gradable_for_fitzpatrick_skin_type",
                "dermatologist_fitzpatrick_skin_type_label",
            ],
        },
    ]

    frames = []
    for group in column_groups:
        df_temp = cases_and_labels_df[group["columns"]].copy()
        df_temp.columns = group["modified_columns"]
        frames.append(df_temp)

    df_final = pd.concat(frames, ignore_index=True)
    df_final = df_final[df_final["image_path"].notna()]

    id_columns = [
        col
        for col in cases_and_labels_df.columns
        if col not in ["image_1_path", "image_2_path", "image_3_path"]
    ]
    image_paths_long = cases_and_labels_df.melt(
        id_vars=id_columns,
        value_vars=["image_1_path", "image_2_path", "image_3_path"],
        var_name="image_path_type",
        value_name="image_path",
    )

    result = pd.merge(df_final, image_paths_long, on="image_path", how="inner")

    selected_columns = [
        "case_id",
        "image_path",
        "shot_type",
        "dermatologist_gradable_for_skin_condition",
        "dermatologist_gradable_for_fitzpatrick_skin_type",
        "dermatologist_fitzpatrick_skin_type_label",
        "source",
        "release",
        "year",
        "age_group",
        "sex_at_birth",
        "fitzpatrick_skin_type",
        "race_ethnicity_american_indian_or_alaska_native",
        "race_ethnicity_asian",
        "race_ethnicity_black_or_african_american",
        "race_ethnicity_hispanic_latino_or_spanish_origin",
        "race_ethnicity_middle_eastern_or_north_african",
        "race_ethnicity_native_hawaiian_or_pacific_islander",
        "race_ethnicity_white",
        "race_ethnicity_other_race",
        "race_ethnicity_prefer_not_to_answer",
        "textures_raised_or_bumpy",
        "textures_flat",
        "textures_rough_or_flaky",
        "textures_fluid_filled",
        "body_parts_head_or_neck",
        "body_parts_arm",
        "body_parts_palm",
        "body_parts_back_of_hand",
        "body_parts_torso_front",
        "body_parts_torso_back",
        "body_parts_genitalia_or_groin",
        "body_parts_buttocks",
        "body_parts_leg",
        "body_parts_foot_top_or_side",
        "body_parts_foot_sole",
        "body_parts_other",
        "condition_symptoms_bothersome_appearance",
        "condition_symptoms_bleeding",
        "condition_symptoms_increasing_size",
        "condition_symptoms_darkening",
        "condition_symptoms_itching",
        "condition_symptoms_burning",
        "condition_symptoms_pain",
        "condition_symptoms_no_relevant_experience",
        "other_symptoms_fever",
        "other_symptoms_chills",
        "other_symptoms_fatigue",
        "other_symptoms_joint_pain",
        "other_symptoms_mouth_sores",
        "other_symptoms_shortness_of_breath",
        "other_symptoms_no_relevant_symptoms",
        "related_category",
        "condition_duration",
        "combined_race",
        "race_ethnicity_two_or_more_after_mitigation",
        "dermatologist_skin_condition_on_label_name",
        "dermatologist_skin_condition_confidence",
        "weighted_skin_condition_label",
        "gradable_for_monk_skin_tone_india",
        "gradable_for_monk_skin_tone_us",
        "monk_skin_tone_label_india",
        "monk_skin_tone_label_us",
    ]
    final_result = result[selected_columns]

    return final_result


def generate_descriptions(final_result):
    """
    Generates descriptions for each row in the DataFrame based on the available data.
    """
    descriptions = []

    for _, row in final_result.iterrows():
        row_description = ""

        if pd.notnull(row["dermatologist_skin_condition_on_label_name"]):
            skin_conditions = (
                row["dermatologist_skin_condition_on_label_name"]
                .strip("[]")
                .replace("'", "")
                .split(", ")
            )
            row_description += f"The dermatologist labeled the skin condition(s) as {', '.join(skin_conditions)}. "

            # Weighted skin condition label
        if pd.notnull(row["weighted_skin_condition_label"]):
            weighted_skin_condition_dict = ast.literal_eval(
                row["weighted_skin_condition_label"]
            )
            weighted_labels = [
                f"{condition}: {weight:.2f}"
                for condition, weight in weighted_skin_condition_dict.items()
            ]
            row_description += (
                f"The weighted skin condition label is: {', '.join(weighted_labels)}. "
            )

        # Condition duration
        if pd.notnull(row["condition_duration"]):
            row_description += f"The condition duration is {row['condition_duration'].replace('_', ' ').lower()}. "

        # Textures
        textures = []
        texture_fields = [
            "textures_raised_or_bumpy",
            "textures_flat",
            "textures_rough_or_flaky",
            "textures_fluid_filled",
        ]
        texture_labels = ["Raised or Bumpy", "Flat", "Rough or Flaky", "Fluid Filled"]

        for field, label in zip(texture_fields, texture_labels):
            if pd.notnull(row[field]) and row[field] == "YES":
                textures.append(label)

        if textures:
            row_description += (
                f"The skin condition has {', '.join(textures)} texture(s). "
            )

        # Body parts
        body_parts = []
        body_part_fields = [
            "body_parts_head_or_neck",
            "body_parts_arm",
            "body_parts_palm",
            "body_parts_back_of_hand",
            "body_parts_torso_front",
            "body_parts_torso_back",
            "body_parts_genitalia_or_groin",
            "body_parts_buttocks",
            "body_parts_leg",
            "body_parts_foot_top_or_side",
            "body_parts_foot_sole",
            "body_parts_other",
        ]
        body_part_labels = [
            "Head or Neck",
            "Arm",
            "Palm",
            "Back of Hand",
            "Torso Front",
            "Torso Back",
            "Genitalia or Groin",
            "Buttocks",
            "Leg",
            "Foot Top or Side",
            "Foot Sole",
            "Other",
        ]

        for field, label in zip(body_part_fields, body_part_labels):
            if pd.notnull(row[field]) and row[field] == "YES":
                body_parts.append(label)

        if body_parts:
            row_description += (
                f"The affected body part(s) are {', '.join(body_parts)}. "
            )

        # Condition symptoms
        condition_symptoms = []
        condition_symptom_fields = [
            "condition_symptoms_bothersome_appearance",
            "condition_symptoms_bleeding",
            "condition_symptoms_increasing_size",
            "condition_symptoms_darkening",
            "condition_symptoms_itching",
            "condition_symptoms_burning",
            "condition_symptoms_pain",
            "condition_symptoms_no_relevant_experience",
        ]
        condition_symptom_labels = [
            "Bothersome Appearance",
            "Bleeding",
            "Increasing Size",
            "Darkening",
            "Itching",
            "Burning",
            "Pain",
            "No Relevant Experience",
        ]

        for field, label in zip(condition_symptom_fields, condition_symptom_labels):
            if pd.notnull(row[field]) and row[field] == "YES":
                condition_symptoms.append(label)

        if condition_symptoms:
            row_description += (
                f"The condition symptoms include {', '.join(condition_symptoms)}. "
            )

        # Other symptoms
        other_symptoms = []
        other_symptom_fields = [
            "other_symptoms_fever",
            "other_symptoms_chills",
            "other_symptoms_fatigue",
            "other_symptoms_joint_pain",
            "other_symptoms_mouth_sores",
            "other_symptoms_shortness_of_breath",
            "other_symptoms_no_relevant_symptoms",
        ]
        other_symptom_labels = [
            "Fever",
            "Chills",
            "Fatigue",
            "Joint Pain",
            "Mouth Sores",
            "Shortness of Breath",
            "No Relevant Symptoms",
        ]

        for field, label in zip(other_symptom_fields, other_symptom_labels):
            if pd.notnull(row[field]) and row[field] == "YES":
                other_symptoms.append(label)

        if other_symptoms:
            row_description += f"Other symptoms include {', '.join(other_symptoms)}. "

        # Related category
        if pd.notnull(row["related_category"]):
            row_description += f"The related category is {row['related_category']}. "

        # Shot type
        if pd.notnull(row["shot_type"]):
            row_description += f"The image shot type is {row['shot_type']}. "

        # Dermatologist gradable for skin condition
        if (
            pd.notnull(row["dermatologist_gradable_for_skin_condition"])
            and row["dermatologist_gradable_for_skin_condition"]
            == "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT"
        ):
            row_description += (
                "The case is gradable for skin condition by the dermatologist. "
            )
        else:
            row_description += (
                "The case is not gradable for skin condition by the dermatologist. "
            )

        # Dermatologist gradable for Fitzpatrick skin type
        if (
            pd.notnull(row["dermatologist_gradable_for_fitzpatrick_skin_type"])
            and row["dermatologist_gradable_for_fitzpatrick_skin_type"] == "YES"
        ):
            row_description += (
                "The case is gradable for Fitzpatrick skin type by the dermatologist. "
            )
        else:
            row_description += "The case is not gradable for Fitzpatrick skin type by the dermatologist. "

        # Dermatologist Fitzpatrick skin type label
        if pd.notnull(row["dermatologist_fitzpatrick_skin_type_label"]):
            row_description += f"The dermatologist labeled the Fitzpatrick skin type as {row['dermatologist_fitzpatrick_skin_type_label']}. "

        # Source, release, and year
        if (
            pd.notnull(row["source"])
            and pd.notnull(row["release"])
            and pd.notnull(row["year"])
        ):
            row_description += f"The source is {row['source']}, the release is {row['release']}, and the year is {row['year']}. "

        # Age group
        if pd.notnull(row["age_group"]):
            row_description += f"The patient's age group is {row['age_group'].replace('AGE_UNKNOWN', 'not known').replace('_', ' ').lower()}. "

        # Sex at birth
        if pd.notnull(row["sex_at_birth"]):
            row_description += f"The patient's sex at birth is {row['sex_at_birth'].replace('OTHER_OR_UNSPECIFIED', 'not specified').lower()}. "

        # Fitzpatrick skin type
        if pd.notnull(row["fitzpatrick_skin_type"]):
            row_description += f"The patient's Fitzpatrick skin type is {row['fitzpatrick_skin_type']}. "

        # Race/ethnicity
        races = []
        race_ethnicity_fields = [
            "race_ethnicity_american_indian_or_alaska_native",
            "race_ethnicity_asian",
            "race_ethnicity_black_or_african_american",
            "race_ethnicity_hispanic_latino_or_spanish_origin",
            "race_ethnicity_middle_eastern_or_north_african",
            "race_ethnicity_native_hawaiian_or_pacific_islander",
            "race_ethnicity_white",
            "race_ethnicity_other_race",
            "race_ethnicity_prefer_not_to_answer",
        ]
        race_ethnicity_labels = [
            "American Indian or Alaska Native",
            "Asian",
            "Black or African American",
            "Hispanic, Latino, or Spanish Origin",
            "Middle Eastern or North African",
            "Native Hawaiian or Pacific Islander",
            "White",
            "Other Race",
            "Prefer Not to Answer",
        ]

        for field, label in zip(race_ethnicity_fields, race_ethnicity_labels):
            if pd.notnull(row[field]) and row[field]:
                races.append(label)

        if races:
            row_description += f"The patient's race/ethnicity is {', '.join(races)}. "

        # Gradable for Monk skin tone (India)
        if pd.notnull(row["gradable_for_monk_skin_tone_india"]):
            row_description += (
                "The case is gradable for Monk skin tone by graders in India. "
            )
        else:
            row_description += (
                "The case is not gradable for Monk skin tone by graders in India. "
            )

        # Gradable for Monk skin tone (US)
        if pd.notnull(row["gradable_for_monk_skin_tone_us"]):
            row_description += (
                "The case is gradable for Monk skin tone by graders in the US. "
            )
        else:
            row_description += (
                "The case is not gradable for Monk skin tone by graders in the US. "
            )

        # Monk skin tone label (India)
        if (
            pd.notnull(row["monk_skin_tone_label_india"])
            and row["monk_skin_tone_label_india"] is not None
        ):
            row_description += f"The Monk skin tone label by graders in India is {row['monk_skin_tone_label_india']}. "

        # Monk skin tone label (US)
        if (
            pd.notnull(row["monk_skin_tone_label_us"])
            and row["monk_skin_tone_label_us"] is not None
        ):
            row_description += f"The Monk skin tone label by graders in the US is {row['monk_skin_tone_label_us']}. "

        descriptions.append(row_description)

    final_result["description"] = descriptions
    return final_result


def generate_question_answers(gold_df):
    """
    Generates question-answer pairs for each row in the DataFrame.
    """
    output = []

    for _, row in gold_df.iterrows():
        question_answer_pairs = []

        if pd.notnull(row["dermatologist_skin_condition_on_label_name"]):
            skin_conditions = (
                row["dermatologist_skin_condition_on_label_name"]
                .strip("[]")
                .replace("'", "")
                .split(", ")
            )
            question1 = "What is this condition?"
            question2 = "What are the skin conditions?"
            dermatologist_skin_condition_on_label_name = f"The dermatologist labeled the skin condition(s) as {', '.join(skin_conditions)}. "
            if (
                row["dermatologist_gradable_for_skin_condition"]
                == "NO_IMAGE_QUALITY_INSUFFICIENT"
            ):
                dermatologist_skin_condition_on_label_name = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
            question_answer_pairs.append(
                {
                    "quetion": question1,
                    "answer": dermatologist_skin_condition_on_label_name,
                }
            )
            question_answer_pairs.append(
                {
                    "quetion": question2,
                    "answer": dermatologist_skin_condition_on_label_name,
                }
            )

            # Weighted skin condition label
            if pd.notnull(row["weighted_skin_condition_label"]):
                weighted_skin_condition_dict = ast.literal_eval(
                    row["weighted_skin_condition_label"]
                )
                weighted_labels = [
                    f"{condition}: {weight:.2f}"
                    for condition, weight in weighted_skin_condition_dict.items()
                ]
                weighted_skin_condition_label = f"The  skin condition label confidence level is: {', '.join(weighted_labels)}. "
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    weighted_skin_condition_label = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What is the skin condition label confidence level?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": weighted_skin_condition_label}
                )

            # Condition duration
            if pd.notnull(row["condition_duration"]):
                condition_duration = f"The condition duration is {row['condition_duration'].replace('_', ' ').lower()}. "
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    condition_duration = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What is the condition duration?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": condition_duration}
                )

            # Textures
            textures = []
            texture_fields = [
                "textures_raised_or_bumpy",
                "textures_flat",
                "textures_rough_or_flaky",
                "textures_fluid_filled",
            ]
            texture_labels = [
                "Raised or Bumpy",
                "Flat",
                "Rough or Flaky",
                "Fluid Filled",
            ]

            for field, label in zip(texture_fields, texture_labels):
                if pd.notnull(row[field]) and row[field] == "YES":
                    textures.append(label)

            if textures:
                texture_description = (
                    f"The skin condition has {', '.join(textures)} texture(s). "
                )
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    texture_description = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What are the textures of the skin condition?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": texture_description}
                )

            # Body parts
            body_parts = []
            body_part_fields = [
                "body_parts_head_or_neck",
                "body_parts_arm",
                "body_parts_palm",
                "body_parts_back_of_hand",
                "body_parts_torso_front",
                "body_parts_torso_back",
                "body_parts_genitalia_or_groin",
                "body_parts_buttocks",
                "body_parts_leg",
                "body_parts_foot_top_or_side",
                "body_parts_foot_sole",
                "body_parts_other",
            ]
            body_part_labels = [
                "Head or Neck",
                "Arm",
                "Palm",
                "Back of Hand",
                "Torso Front",
                "Torso Back",
                "Genitalia or Groin",
                "Buttocks",
                "Leg",
                "Foot Top or Side",
                "Foot Sole",
                "Other",
            ]

            for field, label in zip(body_part_fields, body_part_labels):
                if pd.notnull(row[field]) and row[field] == "YES":
                    body_parts.append(label)

            if body_parts:
                body_description = (
                    f"The affected body part(s) are {', '.join(body_parts)}. "
                )
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    body_description = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What are the affected body parts?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": body_description}
                )

            # Condition symptoms
            condition_symptoms = []
            condition_symptom_fields = [
                "condition_symptoms_bothersome_appearance",
                "condition_symptoms_bleeding",
                "condition_symptoms_increasing_size",
                "condition_symptoms_darkening",
                "condition_symptoms_itching",
                "condition_symptoms_burning",
                "condition_symptoms_pain",
                "condition_symptoms_no_relevant_experience",
            ]
            condition_symptom_labels = [
                "Bothersome Appearance",
                "Bleeding",
                "Increasing Size",
                "Darkening",
                "Itching",
                "Burning",
                "Pain",
                "No Relevant Experience",
            ]

            for field, label in zip(condition_symptom_fields, condition_symptom_labels):
                if pd.notnull(row[field]) and row[field] == "YES":
                    condition_symptoms.append(label)

            if condition_symptoms:
                condition_description = (
                    f"The condition symptoms include {', '.join(condition_symptoms)}. "
                )
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    condition_description = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What are the condition symptoms?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": condition_description}
                )

            # Other symptoms
            other_symptoms = []
            other_symptom_fields = [
                "other_symptoms_fever",
                "other_symptoms_chills",
                "other_symptoms_fatigue",
                "other_symptoms_joint_pain",
                "other_symptoms_mouth_sores",
                "other_symptoms_shortness_of_breath",
                "other_symptoms_no_relevant_symptoms",
            ]
            other_symptom_labels = [
                "Fever",
                "Chills",
                "Fatigue",
                "Joint Pain",
                "Mouth Sores",
                "Shortness of Breath",
                "No Relevant Symptoms",
            ]

            for field, label in zip(other_symptom_fields, other_symptom_labels):
                if pd.notnull(row[field]) and row[field] == "YES":
                    other_symptoms.append(label)

            if other_symptoms:
                other_symptom_description = (
                    f"Some of the symptoms include {', '.join(other_symptoms)}. "
                )
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    other_symptom_description = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What are some of the symptoms?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": other_symptom_description}
                )

            # Related category
            if pd.notnull(row["related_category"]):
                related_cat_description = (
                    f"The related category is {row['related_category']}. "
                )
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    related_cat_description = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What is the related category?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": related_cat_description}
                )

            # # Shot type
            # if pd.notnull(row["shot_type"]):
            #     row_description += f"The image shot type is {row['shot_type']}. "

            # Dermatologist gradable for skin condition
            if (
                pd.notnull(row["dermatologist_gradable_for_skin_condition"])
                and row["dermatologist_gradable_for_skin_condition"]
                == "DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT"
            ):
                grade_description = (
                    "The case is gradable for skin condition by the dermatologist. "
                )
            else:
                grade_description = (
                    "The case is not gradable for skin condition by the dermatologist. "
                )

            question1 = "Is the case gradable for skin condition by the dermatologist?"
            question_answer_pairs.append(
                {"quetion": question1, "answer": grade_description}
            )

            # Race/ethnicity
            races = []
            race_ethnicity_fields = [
                "race_ethnicity_american_indian_or_alaska_native",
                "race_ethnicity_asian",
                "race_ethnicity_black_or_african_american",
                "race_ethnicity_hispanic_latino_or_spanish_origin",
                "race_ethnicity_middle_eastern_or_north_african",
                "race_ethnicity_native_hawaiian_or_pacific_islander",
                "race_ethnicity_white",
                "race_ethnicity_other_race",
                "race_ethnicity_prefer_not_to_answer",
            ]
            race_ethnicity_labels = [
                "American Indian or Alaska Native",
                "Asian",
                "Black or African American",
                "Hispanic, Latino, or Spanish Origin",
                "Middle Eastern or North African",
                "Native Hawaiian or Pacific Islander",
                "White",
                "Other Race",
                "Prefer Not to Answer",
            ]

            for field, label in zip(race_ethnicity_fields, race_ethnicity_labels):
                if pd.notnull(row[field]) and row[field]:
                    races.append(label)

            if races:
                race_description = (
                    f"The patient's race/ethnicity is {', '.join(races)}. "
                )
                if (
                    row["dermatologist_gradable_for_skin_condition"]
                    == "NO_IMAGE_QUALITY_INSUFFICIENT"
                ):
                    race_description = "The image alone is not sufficient to determine the skin condition. Please consult a dermatologist or your healthcare provider."
                question1 = "What is the patient's race?"
                question_answer_pairs.append(
                    {"quetion": question1, "answer": race_description}
                )

            output.append(question_answer_pairs)

        gold_df["description"] = output
        return gold_df


def create_final_output(gold_df):
    """
    Creates the final output in the required format for the task.
    """
    final_output = []
    for i, j in gold_df.iterrows():
        for item in j["description"]:
            formatted_answers = item["answer"]

            data = {
                "id": uuid.uuid4().hex,
                "image": "/root/viVQA-voice-assistant/llava/" + j["image_path"],
                "conversations": [
                    {
                        "from": "human",
                        "value": "[INST] <image>\n" + item["quetion"] + "[/INST]",
                    },
                    {"from": "gpt", "value": formatted_answers},
                ],
            }
            final_output.append(data)

    return final_output


def save_output_to_file(final_output, output_file_path):
    """
    Saves the final output to a JSONL file.
    """
    with open(output_file_path, "w") as outfile:
        for entry in final_output:
            json.dump(entry, outfile)
            outfile.write("\n")


def copy_image_files(gold_df, target_dir):
    """
    Copies image files referenced in the DataFrame to a specified target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for index, row in gold_df.iterrows():
        source_path = row["image_path"]
        target_path = os.path.join(target_dir, os.path.basename(source_path))

        try:
            shutil.copy(source_path, target_path)
        except Exception as e:
            print(f"Failed to copy {source_path} to {target_path}: {e}")

    print("Files have been copied/moved successfully.")


# Example usage:
directory_path = "/Users/prajwalvijendra/dev/skinformatics/dataset"
cases_csv = "/Users/prajwalvijendra/dev/skinformatics/dataset/scin_cases.csv"
labels_csv = "/Users/prajwalvijendra/dev/skinformatics/dataset/scin_labels.csv"

list_files(directory_path)

cases_df = initialize_df_with_metadata(cases_csv)
cases_and_labels_df = augment_metadata_with_labels(cases_df, labels_csv)
print(len(cases_and_labels_df))

final_result = process_dataframe(cases_and_labels_df)
final_result = generate_descriptions(final_result)
gold_df = final_result[final_result["shot_type"] == "CLOSE_UP"]
gold_df = generate_question_answers(gold_df)
final_output = create_final_output(gold_df)
save_output_to_file(final_output, "final_output.jsonl")
copy_image_files(gold_df, "dataset2")
