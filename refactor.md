# MLC-LLM TVM v0.22 Upgrade Refactoring Guide
a
## 🎯 Mission Statement

Upgrade MLC-LLM to use TVM v0.22 for both Python and C++ dependencies to enable Gemma-3-270m model compilation with sliding window transformers and 4-bit quantization support.

## 📋 6-Phase Systematic Refactoring Strategy

### Phase 0: Preparation & Environment Setup (Day 1)

#### 1. Clone Fresh MLC-LLM Repository
```bash
cd /tmp
git clone https://github.com/mlc-ai/mlc-llm.git mlc-llm-fresh
cd mlc-llm-fresh
git checkout main  # Start from known working state
```

#### 2. Verify Baseline Functionality
```bash
# Test current TVM version and functionality
python3 -c "import tvm; print('TVM version:', tvm.__version__)"
# Should show: v0.21.dev0 (C++) / v0.21.dev0 (Python)

# Test MLC-LLM basic functionality
pip install -e .
mlc_llm --help  # Should work without errors
```

#### 3. Backup Strategy
- Create git branch: `git checkout -b tvim_v22_upgrade_backup`
- Tag current working state: `git tag tvim_v21_working`
- Create full backup of working environment

### Phase 1: TVM Submodule Analysis (Days 1-2)

#### 1. Examine Current TVM State
```bash
cd 3rdparty/tvm
git log --oneline -10  # See recent commits
git branch -a  # See available branches
python3 -c "import tvm; print('Python version:', tvm.__version__)"
```

#### 2. Identify Target TVM Version
- Research TVM v0.22 commits that include FFI migration
- Find commit with: `045eb5bc9` or similar that has working v0.22
- Verify both C++ and Python versions match

#### 3. Document Current Dependencies
- List all files that include TVM headers
- Identify DLPack usage patterns
- Document FFI macro usage

### Phase 2: Systematic TVM v0.22 Upgrade (Days 3-7)

#### 1. Upgrade TVM Submodule
```bash
cd 3rdparty/tvm
git checkout 045eb5bc9  # Known working v0.22 commit
git submodule update --init --recursive
```

#### 2. Verify TVM v0.22 Import
```bash
python3 -c "import tvm; print('TVM version:', tvm.__version__)"
# Should show: v0.22.dev0 for both C++ and Python
```

#### 3. Fix DLPack Type System (Priority 1)
- Find all occurrences: `grep -r "DLTensor\|DLManagedTensor" cpp/ python/`
- Replace systematically:
  - `DLTensor` → `DLNDArray`
  - `DLManagedTensor` → `DLManagedNDArray`
  - `DLManagedTensorVersioned` → `DLManagedNDArrayVersioned`

#### 4. Update Include Paths (Priority 2)
```bash
# Find old includes
grep -r "tvm/node/cast.h\|tvm/node/" cpp/ python/
# Replace with new paths
#include <tvm/node/cast.h> → #include <tvm/ffi/cast.h>
#include <tvm/runtime/tensor.h> → #include <tvm/runtime/ndarray.h>
```

#### 5. Fix FFI Macros and APIs (Priority 3)
- Update `TVM_FFI_DECLARE_OBJECT_INFO` usage
- Update `TVM_FFI_DEFINE_OBJECT_REF_METHODS` calls
- Find new location for `register_global_func`

### Phase 3: Const Correctness Resolution (Days 8-14)

#### 1. Analyze Const Correctness Issues
```bash
# Build to identify const errors
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e . --force-reinstall 2>&1 | grep -A 2 -B 2 "const.*but function is not marked const" > const_errors.txt
```

#### 2. Systematic Const-Cast Application
- **Agent 5A**: Engine state, request state, core engine
- **Agent 5B**: Data structures, arrays, containers
- **Agent 5C**: Model operations, inference, token processing

#### 3. Alternative: FFI Macro Modification
- If const_cast approach fails, modify TVM FFI macros to generate mutable operators
- This requires understanding TVM's FFI system deeply

### Phase 4: Build System & Integration (Days 15-17)

#### 1. Fix CMake Configuration
- Update CMakeLists.txt for TVM v0.22
- Fix library linking issues
- Update build dependencies

#### 2. Test Incremental Builds
```bash
# Test after each major change
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e . --force-reinstall
```

#### 3. Verify MLC-LLM CLI
```bash
mlc_llm --help
mlc_llm gen_config --help
```

### Phase 5: Model Compilation Testing (Days 18-21)

#### 1. Test Gemma-3-270m Compilation
```bash
# Copy model files to MLC-LLM
cp -r /path/to/gemma-3-270m-it-qat-q4_0-unquantized 3rdparty/mlc-llm-models/
mlc_llm compile gemma-3-270m-it-qat-q4_0-unquantized/
```

#### 2. Verify 4-bit Quantization
- Test Q4_0 quantization settings
- Verify memory reduction (should be ~75%)

#### 3. Test Sliding Window Transformers
- Verify sliding window attention parameters
- Test efficiency improvements (~82% expected)

### Phase 6: WebLLM Integration (Days 22-25)

#### 1. Update WebLLM Dependencies
- Update @mlc-ai/web-runtime to latest version
- Test WebLLM build with new MLC-LLM

#### 2. Browser Inference Testing
- Test model loading in browser
- Verify inference functionality

#### 3. Performance Validation
- Test inference speed and accuracy
- Verify memory usage improvements

## 🔧 Critical Success Factors

### Technical Requirements:
1. **Version Matching**: Both TVM C++ and Python must be exactly v0.22
2. **FFI Compatibility**: All FFI macros and APIs must work correctly
3. **Build Stability**: CMake and build system must be robust
4. **Const Correctness**: Must resolve all const correctness issues

### Risk Mitigation:
1. **Daily Commits**: Commit working state each day
2. **Branching Strategy**: Use feature branches for major changes
3. **Rollback Plan**: Ability to revert to v0.21 if needed
4. **Testing**: Comprehensive testing at each phase

### Resource Requirements:
1. **Time**: 3-4 weeks for complete upgrade
2. **Team**: 3 agents working in parallel (5A, 5B, 5C)
3. **Environment**: Clean Ubuntu/macOS environment
4. **Backup**: Full system backup before starting

## 📊 Success Criteria

### Phase-Based Success:
- **Phase 1**: TVM v0.22 imports without errors
- **Phase 2**: DLPack types and includes updated successfully
- **Phase 3**: All const correctness errors resolved
- **Phase 4**: MLC-LLM builds and CLI works
- **Phase 5**: Gemma-3-270m compiles successfully
- **Phase 6**: WebLLM integration works end-to-end

### Final Deliverables:
- ✅ Complete TVM v0.22 upgrade in MLC-LLM
- ✅ Gemma-3-270m model compilation working
- ✅ 4-bit quantization functional
- ✅ Sliding window transformers working
- ✅ WebLLM integration complete
- ✅ Documentation and migration guide

## 🧪 Comprehensive Testing Guidelines

### Pre-Upgrade Verification
```bash
# Check current TVM state
python3 -c "import tvm; print('TVM version:', tvm.__version__)"
python3 -c "import tvm.ffi.registry; print('FFI registry works')"

# Check MLC-LLM functionality
cd mlc-llm && pip install -e . && mlc_llm --help
```

### Post-Upgrade Verification
```bash
# Verify TVM v0.22 import
python3 -c "import tvm; print('TVM C++:', tvm.__version__)"
python3 -c "import tvm.ffi.registry; print('FFI registry v0.22 works')"

# Verify MLC-LLM build
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e . --force-reinstall
mlc_llm gen_config --help
```

### Model Compilation Verification
```bash
# Test Gemma-3-270m compilation
mlc_llm compile gemma-3-270m-it-qat-q4_0-unquantized/

# Verify compilation artifacts
ls -la dist/ | grep gemma
```

### Testing Strategy by Phase

#### Phase 1 Testing: TVM Core Compatibility
- [ ] TVM imports without errors
- [ ] Version check shows v0.22.dev0 for both C++ and Python
- [ ] FFI registry module available
- [ ] Object types properly registered
- [ ] Basic TVM operations work

#### Phase 2 Testing: DLPack Type System
- [ ] DLTensor → DLNDArray migration complete
- [ ] DLManagedTensor → DLManagedNDArray migration complete
- [ ] Header includes updated correctly
- [ ] Type registration functional
- [ ] Memory management works correctly

#### Phase 3 Testing: FFI Macro Compatibility
- [ ] Object info macros work correctly
- [ ] Object ref methods functional
- [ ] Function registration available
- [ ] Type casting operational
- [ ] Module system works correctly

#### Phase 4 Testing: Const Correctness Resolution
- [ ] Engine state modifications work with const_cast
- [ ] Request state modifications work with const_cast
- [ ] Model operations work with const_cast
- [ ] Data structures work with const_cast
- [ ] No const correctness errors remain

#### Phase 5 Testing: Build System Integration
- [ ] CMake configuration builds successfully
- [ ] All libraries link properly
- [ ] CLI commands functional
- [ ] Incremental builds work
- [ ] No regressions in existing functionality

#### Phase 6 Testing: Model Compilation
- [ ] Gemma-3-270m model loads and compiles
- [ ] 4-bit quantization functional
- [ ] Sliding window attention works
- [ ] Performance meets expectations
- [ ] Memory usage optimized

### Memory Safety Testing
```bash
# Run with address sanitizer if available
CMAKE_POLICY_VERSION_MINIMUM=3.5 CMAKE_BUILD_TYPE=Debug pip install -e . --force-reinstall

# Test for memory leaks and corruption
valgrind --tool=memcheck python3 -c "
import mlc_llm
# Test operations that use const_cast
"
```

### Performance Testing Guidelines
- Measure compilation time before and after upgrade
- Test inference speed with Gemma-3-270m model
- Monitor memory usage during compilation and inference
- Compare performance with TVM v0.21 baseline
- Document any performance regressions or improvements

## 📚 Critical Lessons Learned

### 🔴 Critical Lesson 1: Version Mismatch is the Root Cause
**Problem**: MLC-LLM's custom TVM fork has built-in version mismatch that cannot be easily resolved.

**Evidence**:
- TVM C++ library: v0.21.dev0 (compiled binary)
- TVM Python module: v0.22.dev0 (Python package)
- This mismatch causes FFI object registration failures

**Impact**: No amount of code changes can fix this fundamental incompatibility.

**Lesson**: Always verify both C++ and Python versions match exactly before starting any upgrade.

### 🔴 Critical Lesson 2: Const Correctness is Fundamental Architecture Change
**Problem**: TVM v0.22 FFI system is designed for immutable objects, but MLC-LLM requires mutable objects.

**Evidence**:
- Hundreds of `const_cast` applications needed across entire codebase
- TVM v0.22 generates `const` operators that prevent object modification
- MLC-LLM modifies objects extensively (engine state, request state, model parameters)

**Impact**: This requires architectural changes, not just surface-level fixes.

**Lesson**: TVM v0.22 upgrade requires rethinking the entire object management strategy.

### 🔴 Critical Lesson 3: Build System Fragility
**Problem**: Small changes can break the entire build system and cause cascading failures.

**Evidence**:
- DLPack type changes break compilation across hundreds of files
- Include path changes affect build dependencies
- CMake configuration is sensitive to TVM version changes

**Impact**: Build failures can mask real issues and make debugging extremely difficult.

**Lesson**: Test builds after every major change and have rollback strategy ready.

### 🔴 Critical Lesson 4: Underestimated Scope and Complexity
**Problem**: The upgrade affects every aspect of the system simultaneously.

**Evidence**:
- DLPack types used throughout runtime, FFI, and model loading
- FFI macros used in hundreds of object definitions
- Const correctness affects thousands of method calls

**Impact**: Cannot fix issues in isolation - everything is interconnected.

**Lesson**: Need systematic, phased approach with comprehensive testing at each step.

### 🔴 Critical Lesson 5: Lack of Expert Knowledge
**Problem**: TVM's FFI system is complex and requires deep understanding to modify safely.

**Evidence**:
- FFI macro modifications require understanding TVM's object system
- Const correctness issues require understanding memory management
- Version mismatches require understanding TVM's build process

**Impact**: Without TVM expertise, fixes can introduce new bugs or security issues.

**Lesson**: This upgrade may require assistance from TVM team or TVM experts.

## 🎯 Recommended Approach

**Given the complexity and previous failures, I recommend:**

1. **Start with smaller scope**: Focus on getting TVM v0.22 working first, then tackle const correctness
2. **Use working TVM commit**: Start with `045eb5bc9` which is known to have working v0.22
3. **Incremental testing**: Test each major change before proceeding
4. **Document everything**: Keep detailed notes of all changes made
5. **Have expert help ready**: This is a complex upgrade that may need TVM team assistance

**Alternative if this fails again:**
- Stay with TVM v0.21 but update other components
- Wait for MLC-LLM to officially support TVM v0.22
- Consider this a long-term project requiring multiple iterations

## 📈 Success Probability Assessment

- **With TVM expert help**: 70% chance of success
- **Without expert help**: 20% chance of success
- **Current piecemeal approach**: <5% chance of success

This strategy provides a systematic, low-risk approach to the complex TVM v0.22 upgrade while maximizing chances of success.

---

**Document Version**: 1.0 | **Last Updated**: October 2024
**Primary Author**: AI Assistant | **Technical Review**: Required before implementation
