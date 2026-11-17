"""監控 WebSocket 結果流：看間隔、marker_id 連續性、有無斷流。

診斷「前端偵測框閃爍/跳動」時用。它會告訴你：
  · 結果間隔（中位/最大）—— 超過前端 maxAgeMs(700ms) 就會沒框可畫
  · marker_id 是否連續 —— 倒退代表 ID 繞回，跳號代表掉結果
  · 有無斷流 —— 伺服器關連線或推播停頓

⚠️ 注意這是「單台相機」的視角。Redis stream 是多台交錯寫入的，
   整體看起來很穩不代表單台夠快 —— 實際踩過：整體中位 258ms 看似正常，
   但單台其實是 362ms（兩台交錯），抖動時就超過 700ms 門檻。

用法：python tools/watch_ws.py [camera_id] [秒數]
      在跑著 output-api 的機器上執行（預設連 localhost:18080）
"""
import json, socket, base64, os, struct, time, http.client, sys, statistics

CAM = sys.argv[1] if len(sys.argv) > 1 else "cam156"
DUR = int(sys.argv[2]) if len(sys.argv) > 2 else 120

key = base64.b64encode(os.urandom(16)).decode()
c = http.client.HTTPConnection('localhost', 18080, timeout=10)
c.putrequest('GET', '/ws', skip_host=True)
c.putheader('Host', 'localhost:18080'); c.putheader('Upgrade', 'websocket')
c.putheader('Connection', 'Upgrade'); c.putheader('Sec-WebSocket-Key', key)
c.putheader('Sec-WebSocket-Version', '13'); c.endheaders()
r = c.getresponse()
print(f"握手 {r.status}")
s = c.sock; s.settimeout(5)

recs = []          # (recv_time, camera, marker_id, n_det)
buf = b""
t0 = time.time()
gaps = []
last_recv = None

def send_pong(payload=b""):
    """回應伺服器的 ping（opcode 0xA），不回會在 40 秒後被踢掉。"""
    hdr = bytes([0x8A, 0x80 | len(payload)]) + b"\x00\x00\x00\x00"
    s.send(hdr + payload)

def parse(data):
    global buf
    buf += data
    out = []
    while len(buf) >= 2:
        b1, b2 = buf[0], buf[1]
        op = b1 & 0x0F
        ln = b2 & 127
        off = 2
        if ln == 126:
            if len(buf) < 4: break
            ln = struct.unpack('>H', buf[2:4])[0]; off = 4
        elif ln == 127:
            if len(buf) < 10: break
            ln = struct.unpack('>Q', buf[2:10])[0]; off = 10
        if len(buf) < off + ln: break
        frame = buf[off:off+ln]
        buf = buf[off+ln:]
        if op == 0x9:      # ping → 回 pong
            send_pong(frame)
        elif op == 0x1:    # text
            out.append(frame)
    return out

print(f"監控 {CAM} {DUR} 秒…")
while time.time() - t0 < DUR:
    try:
        d = s.recv(65536)
    except socket.timeout:
        gap = time.time() - (last_recv or t0)
        print(f"  ⚠ {gap:.1f}s 沒收到任何訊息")
        continue
    except Exception as e:
        print(f"  ✗ 連線中斷: {e}")
        break
    if not d:
        print("  ✗ 伺服器關閉連線")
        break
    for payload in parse(d):
        try:
            j = json.loads(payload)
        except Exception:
            continue
        if 'ref_quads' in j:
            print(f"  [{time.time()-t0:6.1f}s] 收到 ref_quads（連線初始化）")
            continue
        if j.get('camera_id') != CAM:
            continue
        now = time.time()
        if last_recv:
            gaps.append(now - last_recv)
        last_recv = now
        recs.append((now - t0, j.get('camera_id'), j.get('marker_id'),
                     len(j.get('detections') or [])))

print(f"\n===== {CAM} 統計（{DUR}s）=====")
print(f"收到 {len(recs)} 筆結果")
if gaps:
    gaps_ms = [g*1000 for g in gaps]
    print(f"間隔: 中位 {statistics.median(gaps_ms):.0f}ms  平均 {statistics.mean(gaps_ms):.0f}ms  "
          f"最大 {max(gaps_ms):.0f}ms  最小 {min(gaps_ms):.0f}ms")
    big = [(i, g) for i, g in enumerate(gaps_ms) if g > 1000]
    print(f"間隔 >1 秒的次數: {len(big)}")
    for i, g in big[:10]:
        print(f"   第 {i} 筆後停了 {g/1000:.1f}s（約 t={recs[i][0]:.0f}s）")
mids = [m for _, _, m, _ in recs if m is not None]
if len(mids) > 1:
    jumps = [mids[i+1]-mids[i] for i in range(len(mids)-1)]
    print(f"marker_id: {mids[0]} → {mids[-1]}")
    odd = [(i, j) for i, j in enumerate(jumps) if j <= 0 or j > 60]
    print(f"不連續(倒退或跳>60)的次數: {len(odd)}")
    for i, j in odd[:10]:
        print(f"   第 {i} 筆: {mids[i]} → {mids[i+1]} (差 {j})，t={recs[i][0]:.0f}s")
