import serial
import csv
import time
import os

SERIAL_PORT = "/dev/ttyUSB0"   # 안 되면 /dev/ttyACM0 로 바꾸기
BAUD_RATE = 115200

SAVE_FILE = "/home/qqqq/emg_dataset_5ch.csv"

gesture_names = {
    0: "Rest",
    1: "Fist",
    2: "Wrist Flexion",
    3: "Wrist Extension",
    4: "Radial Deviation",
    5: "Ulnar Deviation",
}

def main():
    print("Arduino 5채널 CSV 저장 시작")
    print("저장 파일:", SAVE_FILE)

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)

    header_written = False

    with open(SAVE_FILE, "w", newline="") as f:
        writer = csv.writer(f)

        while True:
            try:
                line = ser.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                # 아두이노 안내문 무시
                if line.startswith("#"):
                    print(line)
                    continue

                # 헤더 줄 무시하고 우리가 직접 헤더 저장
                if line.startswith("time_ms"):
                    if not header_written:
                        writer.writerow([
                            "time_ms",
                            "label",
                            "motion",
                            "ch0",
                            "ch1",
                            "ch2",
                            "ch3",
                            "ch4"
                        ])
                        header_written = True
                        print("CSV 헤더 저장됨")
                    continue

                parts = line.split(",")

                # 현재 아두이노 5채널 형식:
                # time_ms,label,ch0,ch1,ch2,ch3,ch4
                if len(parts) == 7:
                    try:
                        time_ms = int(parts[0])
                        label = int(parts[1])

                        ch0 = int(parts[2])
                        ch1 = int(parts[3])
                        ch2 = int(parts[4])
                        ch3 = int(parts[5])
                        ch4 = int(parts[6])

                        motion = gesture_names.get(label, "Unknown")

                        if not header_written:
                            writer.writerow([
                                "time_ms",
                                "label",
                                "motion",
                                "ch0",
                                "ch1",
                                "ch2",
                                "ch3",
                                "ch4"
                            ])
                            header_written = True
                            print("CSV 헤더 저장됨")

                        writer.writerow([
                            time_ms,
                            label,
                            motion,
                            ch0,
                            ch1,
                            ch2,
                            ch3,
                            ch4
                        ])
                        f.flush()

                        print(f"{time_ms},{label},{motion},{ch0},{ch1},{ch2},{ch3},{ch4}")

                    except ValueError:
                        print("무시된 줄:", line)
                        continue

                else:
                    print("형식 안 맞아서 무시:", line)

            except KeyboardInterrupt:
                print()
                print("저장 종료")
                print("파일 위치:", SAVE_FILE)
                break

    ser.close()

if __name__ == "__main__":
    main()
