# Add this to debug your dataset
print(f"Dataset length: {len(train_dataset)}")
print(f"Dataset type: {type(train_dataset)}")

# If it's a custom dataset, check the first few items
if len(train_dataset) > 0:
    print(f"First item: {train_dataset[0]}")
    print(f"Sample shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in train_dataset[0]]}")
else:
    print("Dataset is empty!")