package dev.luckyedgelord.wav2lipbench;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.qualcomm.qti.QnnDelegate;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Wav2Lip accelerator benchmark. Runs one real Wav2Lip sample (a speech mel window and a
 * silent one, same face) through every way LiteRT can reach this phone's hardware, and reports
 * speed plus whether the output still follows the audio (the WebGPU path silently did not).
 */
public class MainActivity extends Activity {
    private static final String MODEL_INT8 = "wav2lip_gan_b1_int8.tflite";   // full INT8, int8 I/O
    private static final String MODEL_FLOAT = "wav2lip_gan_b1_int8w.tflite"; // INT8 weights, float I/O
    private static final int IMG = 96, RUNS = 8;

    private final StringBuilder log = new StringBuilder();
    private TextView out;
    private Button runBtn;
    private float[] mel, face, expect, expectSilent, silent;
    private double refAudioEffect;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        int pad = dp(16);
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(pad, pad + dp(24), pad, pad);
        col.setBackgroundColor(Color.parseColor("#07060d"));

        TextView title = new TextView(this);
        title.setText("WAV2LIP // ACCELERATOR BENCH");
        title.setTextColor(Color.parseColor("#05d9e8"));
        title.setTextSize(22);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        col.addView(title);

        TextView sub = new TextView(this);
        sub.setText("Runs one real Wav2Lip frame on CPU, GPU, NNAPI and Qualcomm QNN (DSP/GPU). "
                + "Reports ms/frame and whether the mouth still follows the audio.");
        sub.setTextColor(Color.parseColor("#8d87ad"));
        sub.setPadding(0, dp(6), 0, dp(12));
        col.addView(sub);

        runBtn = button("RUN BENCHMARK", "#f9f002", "#07060d");
        runBtn.setOnClickListener(v -> start());
        col.addView(runBtn);
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        Button share = button("SHARE", "#0f0d1a", "#05d9e8"), copy = button("COPY", "#0f0d1a", "#ff2a6d");
        share.setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_SEND).setType("text/plain").putExtra(Intent.EXTRA_TEXT, log.toString());
            startActivity(Intent.createChooser(i, "Share results"));
        });
        copy.setOnClickListener(v -> ((ClipboardManager) getSystemService(CLIPBOARD_SERVICE))
                .setPrimaryClip(ClipData.newPlainText("wav2lip bench", log.toString())));
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(48), 1f);
        half.setMargins(0, dp(8), dp(4), 0);
        row.addView(share, half);
        LinearLayout.LayoutParams half2 = new LinearLayout.LayoutParams(0, dp(48), 1f);
        half2.setMargins(dp(4), dp(8), 0, 0);
        row.addView(copy, half2);
        col.addView(row);

        out = new TextView(this);
        out.setTypeface(Typeface.MONOSPACE);
        out.setTextSize(12);
        out.setTextColor(Color.parseColor("#ece9ff"));
        out.setTextIsSelectable(true);
        out.setPadding(0, dp(12), 0, dp(40));
        col.addView(out);

        ScrollView sv = new ScrollView(this);
        sv.setBackgroundColor(Color.parseColor("#07060d"));
        sv.addView(col);
        setContentView(sv);
        deviceInfo();
    }

    private Button button(String text, String bg, String fg) {
        Button btn = new Button(this);
        btn.setText(text);
        btn.setTextColor(Color.parseColor(fg));
        btn.setBackgroundColor(Color.parseColor(bg));
        btn.setTypeface(Typeface.DEFAULT_BOLD);
        btn.setMinHeight(dp(48));
        return btn;
    }

    private int dp(int v) { return Math.round(v * getResources().getDisplayMetrics().density); }

    private synchronized void say(String s) {
        log.append(s).append('\n');
        String text = log.toString();
        runOnUiThread(() -> out.setText(text));
    }

    private void deviceInfo() {
        say("Device: " + Build.MANUFACTURER + " " + Build.MODEL + " (" + Build.DEVICE + ")");
        if (Build.VERSION.SDK_INT >= 31) say("SoC: " + Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL);
        say("Board/hardware: " + Build.BOARD + " / " + Build.HARDWARE);
        say("Android " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + "), "
                + Runtime.getRuntime().availableProcessors() + " cores, ABI " + Arrays.toString(Build.SUPPORTED_ABIS));
        try {
            int[] v = QnnDelegate.getVersion();
            say("QNN delegate " + Arrays.toString(v) + " capabilities: DSP=" + QnnDelegate.checkCapability(QnnDelegate.Capability.DSP_RUNTIME)
                    + " HTP(int8)=" + QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED)
                    + " HTP(fp16)=" + QnnDelegate.checkCapability(QnnDelegate.Capability.HTP_RUNTIME_FP16)
                    + " GPU=" + QnnDelegate.checkCapability(QnnDelegate.Capability.GPU_RUNTIME));
        } catch (Throwable t) { say("QNN capability check failed: " + t); }
        say("Native lib dir: " + getApplicationInfo().nativeLibraryDir);
        say("");
    }

    private void start() {
        runBtn.setEnabled(false);
        runBtn.setText("RUNNING…");
        new Thread(() -> {
            try { bench(); } catch (Throwable t) { say("FATAL: " + t); }
            runOnUiThread(() -> { runBtn.setEnabled(true); runBtn.setText("RUN AGAIN"); });
        }, "bench").start();
    }

    // ---------------------------------------------------------------- benchmark

    private interface Config { Interpreter.Options make(List<AutoCloseable> owned) throws Exception; }

    private void bench() throws IOException {
        mel = readF32("mel.f32"); face = readF32("face.f32");
        expect = readF32("expect.f32"); expectSilent = readF32("expect_silent.f32");
        silent = new float[mel.length];
        Arrays.fill(silent, -4f);
        refAudioEffect = mouthDiff(expect, expectSilent);
        say(String.format(Locale.US, "Reference (PyTorch): speech vs silence differ by %.4f in the mouth.", refAudioEffect));
        say("Verdict OK = output follows the audio; BROKEN = mouth ignores audio.\n");

        String skel = getApplicationInfo().nativeLibraryDir;
        String cache = getCacheDir().getAbsolutePath();

        run("CPU XNNPACK, 1 thread, full INT8", MODEL_INT8, owned -> new Interpreter.Options().setNumThreads(1));
        run("CPU XNNPACK, 4 threads, full INT8", MODEL_INT8, owned -> new Interpreter.Options().setNumThreads(4));
        run("CPU XNNPACK, 4 threads, INT8 weights (float)", MODEL_FLOAT, owned -> new Interpreter.Options().setNumThreads(4));

        run("NNAPI, full INT8 (no NNAPI CPU fallback)", MODEL_INT8, owned -> {
            NnApiDelegate d = new NnApiDelegate(new NnApiDelegate.Options().setUseNnapiCpu(false)
                    .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED));
            owned.add(d);
            return new Interpreter.Options().setNumThreads(4).addDelegate(d);
        });
        run("NNAPI, float, FP16 allowed", MODEL_FLOAT, owned -> {
            NnApiDelegate d = new NnApiDelegate(new NnApiDelegate.Options().setUseNnapiCpu(false).setAllowFp16(true)
                    .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED));
            owned.add(d);
            return new Interpreter.Options().setNumThreads(4).addDelegate(d);
        });

        run("LiteRT GPU delegate, FP16", MODEL_FLOAT, owned -> {
            GpuDelegate d = new GpuDelegate(new GpuDelegateFactory.Options().setPrecisionLossAllowed(true)
                    .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED));
            owned.add(d);
            return new Interpreter.Options().addDelegate(d);
        });
        run("LiteRT GPU delegate, FP32", MODEL_FLOAT, owned -> {
            GpuDelegate d = new GpuDelegate(new GpuDelegateFactory.Options().setPrecisionLossAllowed(false)
                    .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED));
            owned.add(d);
            return new Interpreter.Options().addDelegate(d);
        });

        run("QNN DSP (Hexagon v66), full INT8", MODEL_INT8, owned -> {
            QnnDelegate.Options o = new QnnDelegate.Options();
            o.setBackendType(QnnDelegate.Options.BackendType.DSP_BACKEND);
            o.setSkelLibraryDir(skel);
            o.setCacheDir(cache);
            o.setDspPerformanceMode(QnnDelegate.Options.DspPerformanceMode.DSP_PERFORMANCE_BURST);
            QnnDelegate d = new QnnDelegate(o);
            owned.add(d);
            return new Interpreter.Options().setNumThreads(4).addDelegate(d);
        });
        run("QNN GPU, float (FP16)", MODEL_FLOAT, owned -> {
            QnnDelegate.Options o = new QnnDelegate.Options();
            o.setBackendType(QnnDelegate.Options.BackendType.GPU_BACKEND);
            o.setCacheDir(cache);
            o.setGpuPrecision(QnnDelegate.Options.GpuPrecision.GPU_PRECISION_FP16);
            QnnDelegate d = new QnnDelegate(o);
            owned.add(d);
            return new Interpreter.Options().setNumThreads(4).addDelegate(d);
        });
        say("HTP (Snapdragon 8 Gen 1+ tensor accelerator) not included in this build; LiteRT NPU path "
                + "is not supported on this SoC.");
        say("\nDone. Tap SHARE or COPY to send these results.");
    }

    private void run(String name, String modelFile, Config cfg) {
        say("▶ " + name);
        List<AutoCloseable> owned = new ArrayList<>();
        Interpreter it = null;
        try {
            long t0 = System.nanoTime();
            it = new Interpreter(mapAsset(modelFile), cfg.make(owned));
            double initMs = (System.nanoTime() - t0) / 1e6;
            float[] a = infer(it, mel);                       // first run (may include compilation)
            float[] s = infer(it, silent);
            for (int i = 0; i < 2; i++) infer(it, mel);       // warm-up
            double[] t = new double[RUNS];
            for (int i = 0; i < RUNS; i++) {
                long st = System.nanoTime();
                infer(it, mel);
                t[i] = (System.nanoTime() - st) / 1e6;
            }
            Arrays.sort(t);
            double err = meanAbs(a, expect), effect = mouthDiff(a, s);
            String verdict = effect > refAudioEffect * 0.3 && err < 0.03 ? "OK"
                    : effect < refAudioEffect * 0.1 ? "BROKEN (ignores audio)" : "SUSPECT";
            say(String.format(Locale.US, "  %s  median %.0f ms/frame (min %.0f, max %.0f), init %.0f ms",
                    verdict, t[RUNS / 2], t[0], t[RUNS - 1], initMs));
            say(String.format(Locale.US, "  error vs reference %.4f, audio effect %.4f (ref %.4f)\n", err, effect, refAudioEffect));
        } catch (Throwable e) {
            say("  FAILED: " + e.getClass().getSimpleName() + ": " + e.getMessage() + "\n");
        } finally {
            if (it != null) it.close();
            for (AutoCloseable c : owned) { try { c.close(); } catch (Exception ignored) { } }
        }
    }

    /** One forward pass; quantizes int8 inputs and dequantizes int8 outputs as needed. */
    private float[] infer(Interpreter it, float[] melIn) {
        Object[] inputs = new Object[it.getInputTensorCount()];
        for (int i = 0; i < inputs.length; i++) {
            Tensor t = it.getInputTensor(i);
            boolean isMel = t.shape()[3] == 1;
            inputs[i] = pack(isMel ? melIn : face, t);
        }
        Tensor ot = it.getOutputTensor(0);
        ByteBuffer ob = ByteBuffer.allocateDirect(ot.numBytes()).order(ByteOrder.nativeOrder());
        Map<Integer, Object> outs = new HashMap<>();
        outs.put(0, ob);
        it.runForMultipleInputsOutputs(inputs, outs);
        ob.rewind();
        float[] r = new float[ot.numElements()];
        if (ot.dataType() == DataType.INT8) {
            float sc = ot.quantizationParams().getScale(); int zp = ot.quantizationParams().getZeroPoint();
            for (int i = 0; i < r.length; i++) r[i] = (ob.get() - zp) * sc;
        } else ob.asFloatBuffer().get(r);
        return r;
    }

    private static ByteBuffer pack(float[] src, Tensor t) {
        ByteBuffer b = ByteBuffer.allocateDirect(t.numBytes()).order(ByteOrder.nativeOrder());
        if (t.dataType() == DataType.INT8) {
            float inv = 1f / t.quantizationParams().getScale(); int zp = t.quantizationParams().getZeroPoint();
            for (float v : src) b.put((byte) Math.max(-128, Math.min(127, Math.round(v * inv) + zp)));
        } else b.asFloatBuffer().put(src);
        b.rewind();
        return b;
    }

    private static double meanAbs(float[] a, float[] b) {
        double s = 0; for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length;
    }

    /** Mean |a-b| over the mouth region of a 96x96x3 image. */
    private static double mouthDiff(float[] a, float[] b) {
        double s = 0; int n = 0;
        for (int y = 60; y < 85; y++) for (int x = 30; x < 66; x++) for (int c = 0; c < 3; c++) {
            int i = (y * IMG + x) * 3 + c; s += Math.abs(a[i] - b[i]); n++;
        }
        return s / n;
    }

    private MappedByteBuffer mapAsset(String name) throws IOException {
        try (AssetFileDescriptor fd = getAssets().openFd(name);
             FileInputStream in = new FileInputStream(fd.getFileDescriptor())) {
            return in.getChannel().map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    private float[] readF32(String name) throws IOException {
        try (InputStream in = getAssets().open(name)) {
            byte[] bytes = new byte[in.available()];
            int off = 0, n;
            while (off < bytes.length && (n = in.read(bytes, off, bytes.length - off)) > 0) off += n;
            float[] f = new float[bytes.length / 4];
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(f);
            return f;
        }
    }
}
