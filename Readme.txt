====================================================================
 Table 4 三方法标准化基准 - 运行说明
====================================================================

所有命令在本包根目录（含 01_Datasets ~ 07_Reproduction 的目录）下执行；
脚本均按包根相对路径解析，无需复制文件到其他位置。

完整命令（含 Windows PowerShell 版）见：
    07_Reproduction/reproduction_commands.txt

--------------------------------------------------------------------
 快速运行（macOS / Linux）
--------------------------------------------------------------------
 1. 获取两个第三方工具
    git clone https://github.com/TeamErlich/dna-fountain.git \
        02_Software/DNA-Fountain/dna-fountain-master
    git clone https://github.com/DNAStorageToolkit/DNAStorageToolkit.git \
        02_Software/DNA-Storage-Toolkit/DNAStorageToolkit-main
    cp -R 02_Software/DNA-Storage-Toolkit/bits \
          02_Software/DNA-Storage-Toolkit/DNAStorageToolkit-main/1-encoding-decoding/bits

 2. 建立 Python 环境
    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -r 05_Environment/requirements.txt

 3. 编译编解码器
    (cd 02_Software/DNA-Fountain/dna-fountain-master && python setup.py build_ext --inplace)
    cd 02_Software/DNA-Storage-Toolkit/DNAStorageToolkit-main/1-encoding-decoding
    python -m ziglang c++ -std=c++17 -O2 -I. encoder.cpp -o encoder
    python -m ziglang c++ -std=c++17 -O2 -I. decoder.cpp -o decoder
    cd -

 4. 校验数据集
    (cd 01_Datasets/benchmark_dataset && shasum -a 256 -c SHA256SUMS.txt)

 5. 运行基准（N=10；快速冒烟测试可用 --runs 1）
    python 04_Benchmark/benchmark_scripts/run_benchmark.py --runs 10

 6. 固定源码哈希，生成 Table 4
    python 04_Benchmark/benchmark_scripts/add_source_hashes.py
    python 04_Benchmark/benchmark_scripts/make_table4.py

 7. 恢复图（mandrill + 全部 20 图，命令见 STEP 6/7）
    python 04_Benchmark/benchmark_scripts/recover_outputs.py

定量输出位于 04_Benchmark/results/main/（CSV/JSON/environment.txt/
summary.md/table4.md，重新跑分生成）；
20 图恢复证据已提交在 06_Result/<image_id>/。
