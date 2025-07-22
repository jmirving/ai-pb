# Pre-Training Checklist

Before running a training job, ensure the following steps and checks are complete:

## 1. Dataset Filtering and Structure
- [x] Dataset only includes the latest patch (filtered in `DraftDataset`)
- [x] Only team rows (`participantid` 100, 200) are included
- [x] Data is ordered by `seriesid`, `gameid`, and draft event order
- [x] Each sample represents a single draft event (pick/ban)

## 2. DraftProcessor Output
- [ ] `DraftProcessor.process(row)` returns:
    - [ ] `draft_sequence`: List/array of champion indices (length 20, with correct padding)
    - [ ] `target`: The correct champion index for the current event
    - [ ] `already_picked_or_banned`: Set of champions for output masking

## 3. Model Input/Output Alignment
- [x] Model expects input dictionary with `'draft_sequence'`
- [x] Embedding/input layer matches number of champions and sequence length
- [x] Output size matches number of real champions (excluding MISSING)

## 4. DataLoader and Batching
- [x] DataLoader correctly batches samples (default PyTorch collate works for dicts)
- [x] Batch tensors are the correct shape and type

## 5. Data Types and Device
- [x] All tensors are of the correct dtype (e.g., `torch.long` for indices)
- [x] All tensors are moved to the correct device (CPU/GPU) in the training loop

## 6. (Optional) Test Script/Notebook
- [ ] Instantiate the dataset and print a few samples
- [ ] Pass a batch through the model to check for shape/type errors
- [ ] Verify loss and masking logic on a small batch

---

**Complete all items above before starting a full training run to avoid runtime errors and ensure data/model alignment.** 