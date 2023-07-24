from pathlib import Path
import argparse

import torch
import numpy as np

# std::ofstream fs("tensor.bin", std::ios::out | std::ios::binary | std::ios::app);
# fs.write(reinterpret_cast<const char*>(&tensor), sizeof tensor);
# fs.close();

def save_torch_tensor(t: torch.tensor, path=Path("./orig_input.pt")):
  torch.save(t, path)

def load_torch_tensor(path=Path("./orig_input.pt")):
  return torch.load(path)

def advanced_compare(lft, rht, atol=1e-5, rtol=1e-5):
  if len(lft.shape) > 1:
    lft = lft.flatten()
  if len(rht.shape) > 1:
    lft = rht.flatten()
  numel = lft.shape[0]
  assert numel == rht.shape[0]
  counter = 0
  rtols=[rtol]
  for i in range(numel):
    diff = np.abs(lft[i]-rht[i])
    exp_diff = atol + rtol*np.abs(rht[i])
    if diff > exp_diff:
      new_rtol = (diff - atol)/np.abs(rht[i])
      rtols.append(new_rtol)
      print("Elements with index", i, " are not the same left:", lft[i], " right:", rht[i])
      counter = counter + 1
  print("Number of diverged values:", counter, " Percent is", 100*float(counter)/numel,"%")
  max_rtol = np.max(rtols)
  print("Current rtol:", rtol, "Maximum rtol:", max_rtol)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--rtol', type=float, default=5e-3,
                      help="Relative tolerance")
  parser.add_argument('-a', '--atol', type=float, default=1e-6,
                      help="Absolute tolerance")
  parser.add_argument('-w', '--check_weight', default=False, action="store_true",
                      help="Compare weights. Corresponding files are required")

  args = parser.parse_args()

  check_num = 10
  # Load data from Relax model
  np_input = np.fromfile(Path("./relax_input.bin"), dtype="float32")
  np_weight = np.fromfile(Path("./relax_weight.bin"), dtype="float32")
  print("RELAX INPUT TYPE:", np_input.dtype, "SHAPE:", np_input.shape)
  print("RELAX WEIGHT TYPE:", np_weight.dtype, "SHAPE:", np_weight.shape)

  # Load data from original model
  orig_input = load_torch_tensor()
  orig_weight = load_torch_tensor(Path("./orig_weight.pt"))

  orig_np_input = orig_input.numpy()
  orig_np_weight = orig_weight.numpy()
  print("ORIG INPUT TYPE:", orig_np_input.dtype, "SHAPE:", orig_np_input.shape)
  print("ORIG WEIGHT TYPE:", orig_np_weight.dtype, "SHAPE:", orig_np_weight.shape)

  print("Compare inputs")
  print("ORIG INPUT:", orig_np_input[:check_num])
  print("RELAX INPUT:", np_input[:check_num])
  # np.testing.assert_allclose(orig_np_input, np_input, rtol=rtol, atol=atol, verbose=True)
  advanced_compare(orig_np_input, np_input, rtol=args.rtol, atol=args.atol)

  if args.check_weight:
    print("Compare weights")
    orig_np_line = orig_np_weight[0,:]
    print("ORIG WEIGHT:", orig_np_line[:check_num])
    print("RELAX WEIGHT:", np_weight[:check_num])
    np.testing.assert_allclose(orig_np_line, np_weight, rtol=args.rtol, atol=args.atol, verbose=True)

if __name__ == "__main__":
  main()
