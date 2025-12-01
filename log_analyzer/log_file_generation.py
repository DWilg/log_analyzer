with open("sample_logs/big_example.log", "w") as f:
    for i in range(1_000_000):
        if i % 3 == 0:
            f.write(f"2025-12-01 10:{i%60:02d}:00 ERROR Test error {i}\n")
        elif i % 3 == 1:
            f.write(f"2025-12-01 10:{i%60:02d}:00 WARNING Test warning {i}\n")
        else:
            f.write(f"2025-12-01 10:{i%60:02d}:00 INFO Test info {i}\n")