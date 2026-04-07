/*
 * main.c - CLI entry point for voxtral.c
 *
 * Usage: voxtral -d <model_dir> -i <input.wav> [options]
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_audio.h"
#include "voxtral_mic.h"
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <termios.h>
#include <fcntl.h>
#include <sys/select.h>

#define DEFAULT_FEED_CHUNK 16000 /* 1 second at 16kHz */

/* SIGINT handler for clean exit from --from-mic */
static volatile sig_atomic_t mic_interrupted = 0;
static void sigint_handler(int sig) { (void)sig; mic_interrupted = 1; }

static void usage(const char *prog) {
    fprintf(stderr, "voxtral.c — Voxtral Realtime 4B speech-to-text\n\n");
    fprintf(stderr, "Usage: %s -d <model_dir> (-i <input.wav> | --stdin | --from-mic) [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>      Model directory (with consolidated.safetensors, tekken.json)\n");
    fprintf(stderr, "  -i <file>     Input WAV file (16-bit PCM, any sample rate)\n");
    fprintf(stderr, "  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)\n");
    fprintf(stderr, "  --from-mic    Capture from default microphone (macOS only, Ctrl+C to stop)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -I <secs>     Encoder processing interval in seconds (default: 2.0)\n");
    fprintf(stderr, "  --alt <c>     Show alternative tokens within cutoff distance (0.0-1.0)\n");
    fprintf(stderr, "  --monitor     Show non-intrusive symbols inline with output (stderr)\n");
    fprintf(stderr, "  --copy        Copy transcription to clipboard when done (macOS only)\n");
    fprintf(stderr, "  --space       Hold spacebar to record, release to stop (macOS only)\n");
    fprintf(stderr, "  --debug       Debug output (per-layer, per-chunk details)\n");
    fprintf(stderr, "  --silent      No status output (only transcription on stdout)\n");
    fprintf(stderr, "  -h            Show this help\n");
}

/* Drain pending tokens from stream and print to stdout */
static int first_token = 1;
/* Accumulation buffer for --copy mode */
static char *copy_buffer = NULL;
static size_t copy_buffer_len = 0;
static size_t copy_buffer_cap = 0;

static void accum_copy(const char *text) {
    if (!text || !copy_buffer) return;
    size_t len = strlen(text);
    if (copy_buffer_len + len + 1 > copy_buffer_cap) {
        copy_buffer_cap = (copy_buffer_len + len + 1) * 2;
        if (copy_buffer_cap < 4096) copy_buffer_cap = 4096;
        copy_buffer = realloc(copy_buffer, copy_buffer_cap);
    }
    memcpy(copy_buffer + copy_buffer_len, text, len);
    copy_buffer_len += len;
    copy_buffer[copy_buffer_len] = '\0';
}
static float alt_cutoff = -1; /* <0 means disabled */

static void drain_tokens(vox_stream_t *s) {
    if (alt_cutoff < 0) {
        /* Fast path: no alternatives */
        const char *tokens[64];
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                const char *t = tokens[i];
                if (first_token) {
                    while (*t == ' ') t++;
                    first_token = 0;
                }
                fputs(t, stdout);
                accum_copy(t);
            }
            fflush(stdout);
        }
    } else {
        /* Alternatives mode */
        const int n_alt = 3;
        const char *tokens[64 * 3];
        int n;
        while ((n = vox_stream_get_alt(s, tokens, 64, n_alt)) > 0) {
            for (int i = 0; i < n; i++) {
                const char *best = tokens[i * n_alt];
                if (!best) continue;
                /* Check for alternatives */
                int has_alt = 0;
                for (int a = 1; a < n_alt; a++) {
                    if (tokens[i * n_alt + a]) { has_alt = 1; break; }
                }
                if (has_alt) {
                    fputc('[', stdout);
                    for (int a = 0; a < n_alt; a++) {
                        const char *alt = tokens[i * n_alt + a];
                        if (!alt) break;
                        if (a > 0) fputc('|', stdout);
                        const char *t = alt;
                        if (a == 0 && first_token) {
                            while (*t == ' ') t++;
                            first_token = 0;
                        }
                        fputs(t, stdout);
                        if (a == 0) accum_copy(t);
                    }
                    fputc(']', stdout);
                } else {
                    const char *t = best;
                    if (first_token) {
                        while (*t == ' ') t++;
                        first_token = 0;
                    }
                    fputs(t, stdout);
                    accum_copy(t);
                }
            }
            fflush(stdout);
        }
    }
}

/* Feed audio in chunks, printing tokens as they become available.
 * feed_chunk controls granularity: smaller = more responsive token output. */
static int feed_chunk = DEFAULT_FEED_CHUNK;
static void feed_and_drain(vox_stream_t *s, const float *samples, int n_samples) {
    int off = 0;
    while (off < n_samples) {
        int chunk = n_samples - off;
        if (chunk > feed_chunk) chunk = feed_chunk;
        vox_stream_feed(s, samples + off, chunk);
        off += chunk;
        drain_tokens(s);
    }
}

/* Copy text to clipboard using pbcopy (macOS) */
static void copy_to_clipboard(const char *text) {
    if (!text || !*text) return;
    FILE *pbcopy = popen("pbcopy", "w");
    if (pbcopy) {
        fputs(text, pbcopy);
        pclose(pbcopy);
    }
}

/* Terminal raw mode for spacebar detection */
static struct termios orig_termios;
static int tty_raw_mode = 0;

static void tty_set_raw(void) {
    struct termios raw;
    tcgetattr(STDIN_FILENO, &orig_termios);
    raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    tty_raw_mode = 1;
    /* Make stdin non-blocking */
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

static void tty_restore(void) {
    if (tty_raw_mode) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        tty_raw_mode = 0;
    }
}

/* Check if spacebar is currently pressed (non-blocking) */
static int spacebar_is_pressed(void) {
    char c;
    while (read(STDIN_FILENO, &c, 1) > 0) {
        if (c == ' ') return 1;
        if (c == 3 || c == 'q' || c == 'Q') { /* Ctrl+C or q */
            tty_restore();
            exit(0);
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_wav = NULL;
    int verbosity = 1; /* 0=silent, 1=normal, 2=debug */
    int use_stdin = 0;
    int use_mic = 0;
    int use_space = 0;
    int do_copy = 0;
    float interval = -1.0f; /* <0 means use default */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            interval = (float)atof(argv[++i]);
            if (interval <= 0) {
                fprintf(stderr, "Error: -I requires a positive number of seconds\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--alt") == 0 && i + 1 < argc) {
            alt_cutoff = (float)atof(argv[++i]);
            if (alt_cutoff < 0 || alt_cutoff > 1) {
                fprintf(stderr, "Error: --alt requires a value between 0.0 and 1.0\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        } else if (strcmp(argv[i], "--from-mic") == 0) {
            use_mic = 1;
        } else if (strcmp(argv[i], "--monitor") == 0) {
            extern int vox_monitor;
            vox_monitor = 1;
        } else if (strcmp(argv[i], "--copy") == 0) {
            do_copy = 1;
        } else if (strcmp(argv[i], "--space") == 0) {
            use_space = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input_wav && !use_stdin && !use_mic && !use_space)) {
        usage(argv[0]);
        return 1;
    }
    if ((input_wav ? 1 : 0) + use_stdin + use_mic + use_space > 1) {
        fprintf(stderr, "Error: -i, --stdin, --from-mic, and --space are mutually exclusive\n");
        return 1;
    }
    if (use_space && use_stdin) {
        fprintf(stderr, "Error: --space cannot be used with --stdin\n");
        return 1;
    }

    vox_verbose = verbosity;
    vox_verbose_audio = (verbosity >= 2) ? 1 : 0;

    /* Initialize copy buffer if --copy is enabled */
    if (do_copy) {
        copy_buffer_cap = 4096;
        copy_buffer = malloc(copy_buffer_cap);
        copy_buffer[0] = '\0';
    }

#ifdef USE_METAL
    vox_metal_init();
#endif

    /* Load model */
    vox_ctx_t *ctx = vox_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) {
        fprintf(stderr, "Failed to init stream\n");
        vox_free(ctx);
        return 1;
    }
    if (alt_cutoff >= 0)
        vox_stream_set_alt(s, 3, alt_cutoff);
    if (interval > 0) {
        vox_set_processing_interval(s, interval);
        feed_chunk = (int)(interval * VOX_SAMPLE_RATE);
        if (feed_chunk < 160) feed_chunk = 160;
        if (feed_chunk > DEFAULT_FEED_CHUNK) feed_chunk = DEFAULT_FEED_CHUNK;
    }

    /* Enable continuous mode for live sources (auto-restart decoder) */
    if (use_mic || use_stdin)
        vox_stream_set_continuous(s, 1);

    if (use_mic) {
        /* Microphone capture with silence cancellation */
        if (vox_mic_start() != 0) {
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }

        /* Install SIGINT handler for clean Ctrl+C exit */
        struct sigaction sa;
        sa.sa_handler = sigint_handler;
        sa.sa_flags = 0;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, NULL);

        if (vox_verbose >= 1)
            fprintf(stderr, "Listening (Ctrl+C to stop)...\n");

        /* Silence cancellation state */
        #define MIC_WINDOW 160          /* 10ms at 16kHz */
        #define SILENCE_THRESH 0.002f   /* RMS threshold (~-54 dBFS) */
        #define SILENCE_PASS 60         /* pass-through windows (600ms) */
        float mic_buf[4800]; /* 300ms max read */
        int silence_count = 0;
        int was_skipping = 0; /* were we skipping silence? */
        int overbuf_warned = 0;

        while (!mic_interrupted) {
            /* Over-buffer detection */
            int avail = vox_mic_read_available();
            if (avail > 80000) { /* > 5 seconds buffered */
                if (!overbuf_warned) {
                    fprintf(stderr, "Warning: can't keep up, skipping audio\n");
                    overbuf_warned = 1;
                }
                /* Drain all but last ~1 second */
                float discard[4800];
                while (vox_mic_read_available() > 16000)
                    vox_mic_read(discard, 4800);
                silence_count = 0;
                was_skipping = 0;
            } else if (avail < 32000) { /* < 2 seconds: clear warning */
                overbuf_warned = 0;
            }

            int n = vox_mic_read(mic_buf, 4800);
            if (n == 0) {
                usleep(10000); /* 10ms idle sleep */
                continue;
            }

            /* Process in 10ms windows for silence cancellation */
            int off = 0;
            while (off + MIC_WINDOW <= n) {
                /* Compute RMS energy of this window */
                float energy = 0;
                for (int i = 0; i < MIC_WINDOW; i++) {
                    float v = mic_buf[off + i];
                    energy += v * v;
                }
                float rms = sqrtf(energy / MIC_WINDOW);

                if (rms > SILENCE_THRESH) {
                    /* Voice detected */
                    if (was_skipping)
                        was_skipping = 0;
                    vox_stream_feed(s, mic_buf + off, MIC_WINDOW);
                    silence_count = 0;
                } else {
                    /* Silence detected */
                    silence_count++;
                    if (silence_count <= SILENCE_PASS) {
                        /* Short silence: pass through (natural word gap) */
                        vox_stream_feed(s, mic_buf + off, MIC_WINDOW);
                    } else if (!was_skipping) {
                        /* Entering silence: flush buffered audio */
                        was_skipping = 1;
                        vox_stream_flush(s);
                    }
                }
                off += MIC_WINDOW;
            }

            /* Feed any remaining samples (< 1 window) */
            if (off < n)
                vox_stream_feed(s, mic_buf + off, n - off);

            drain_tokens(s);
        }

        vox_mic_stop();
        if (vox_verbose >= 1)
            fprintf(stderr, "\nStopping...\n");
    } else if (use_stdin) {
        /* Read enough to detect WAV vs raw and parse WAV header */
        uint8_t hdr[4096];
        size_t hdr_read = fread(hdr, 1, sizeof(hdr), stdin);
        if (hdr_read < 4) {
            fprintf(stderr, "Not enough data on stdin\n");
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }

        /* Offset into hdr[] where PCM data starts (0 = raw s16le) */
        size_t pcm_offset = 0;

        if (hdr_read >= 44 && memcmp(hdr, "RIFF", 4) == 0 &&
            memcmp(hdr + 8, "WAVE", 4) == 0) {
            /* Parse WAV header to find data chunk */
            int wav_fmt = 0, wav_ch = 0, wav_rate = 0, wav_bits = 0;
            const uint8_t *p = hdr + 12;
            const uint8_t *end = hdr + hdr_read;
            int found_data = 0;
            while (p + 8 <= end) {
                uint32_t csz = (uint32_t)(p[4] | (p[5]<<8) | (p[6]<<16) | (p[7]<<24));
                if (memcmp(p, "fmt ", 4) == 0 && csz >= 16 && p + 8 + csz <= end) {
                    wav_fmt  = p[8] | (p[9]<<8);
                    wav_ch   = p[10] | (p[11]<<8);
                    wav_rate  = p[12] | (p[13]<<8) | (p[14]<<16) | (p[15]<<24);
                    wav_bits = p[22] | (p[23]<<8);
                } else if (memcmp(p, "data", 4) == 0) {
                    pcm_offset = (size_t)(p + 8 - hdr);
                    found_data = 1;
                    break;
                }
                if (p + 8 + csz > end) break;
                p += 8 + csz;
                if (csz & 1) p++;
            }
            if (!found_data || wav_fmt != 1 || wav_bits != 16 || wav_ch < 1) {
                fprintf(stderr, "Invalid WAV on stdin (fmt=%d bits=%d)\n",
                        wav_fmt, wav_bits);
                vox_stream_free(s); vox_free(ctx); return 1;
            }
            if (wav_rate != VOX_SAMPLE_RATE || wav_ch != 1) {
                fprintf(stderr, "WAV stdin streaming requires 16kHz mono "
                        "(got %dHz %dch). Use: ffmpeg -i pipe:0 "
                        "-ar 16000 -ac 1 -f s16le pipe:1\n", wav_rate, wav_ch);
                vox_stream_free(s); vox_free(ctx); return 1;
            }
            if (vox_verbose >= 2)
                fprintf(stderr, "Streaming WAV s16le 16kHz mono from stdin\n");
        } else {
            if (vox_verbose >= 2)
                fprintf(stderr, "Streaming raw s16le 16kHz mono from stdin\n");
        }

        /* Feed any PCM data already in the header buffer */
        size_t pcm_in_hdr = hdr_read - pcm_offset;
        size_t pcm_frames = pcm_in_hdr / 2;
        if (pcm_frames > 0) {
            const int16_t *src = (const int16_t *)(hdr + pcm_offset);
            float fbuf[2048];
            for (size_t i = 0; i < pcm_frames; i++)
                fbuf[i] = src[i] / 32768.0f;
            vox_stream_feed(s, fbuf, (int)pcm_frames);
            drain_tokens(s);
        }

        /* Stream the rest incrementally */
        int16_t raw_buf[4096];
        float fbuf[4096];
        while (1) {
            size_t nread = fread(raw_buf, sizeof(int16_t), 4096, stdin);
            if (nread == 0) break;
            for (size_t i = 0; i < nread; i++)
                fbuf[i] = raw_buf[i] / 32768.0f;
            vox_stream_feed(s, fbuf, (int)nread);
            drain_tokens(s);
        }
    } else if (use_space) {
        /* Spacebar hold-to-record mode - loops until Ctrl+C */
        if (vox_mic_start() != 0) {
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }

        /* Setup terminal for raw mode */
        tty_set_raw();

        /* Install atexit handler to restore terminal */
        atexit(tty_restore);

        /* Install SIGINT handler for clean exit */
        struct sigaction sa;
        sa.sa_handler = sigint_handler;
        sa.sa_flags = 0;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, NULL);

        if (vox_verbose >= 1)
            fprintf(stderr, "Hold SPACE to record, release to stop (Ctrl+C to exit)...\n");

        /* Main loop - keep recording until interrupted */
        while (!mic_interrupted) {
            /* Wait for spacebar press */
            if (vox_verbose >= 1)
                fprintf(stderr, "\nWaiting for spacebar...\n");

            while (!mic_interrupted && !spacebar_is_pressed()) {
                usleep(10000); /* 10ms */
            }

            if (mic_interrupted)
                break;

            if (vox_verbose >= 1)
                fprintf(stderr, "Recording...\n");

            float mic_buf[4800]; /* 300ms max read */
            int is_recording = 1;

            while (!mic_interrupted && is_recording) {
                /* Check if spacebar is still pressed */
                if (!spacebar_is_pressed()) {
                    /* Spacebar released - stop recording */
                    is_recording = 0;
                    break;
                }

                int n = vox_mic_read(mic_buf, 4800);
                if (n > 0) {
                    vox_stream_feed(s, mic_buf, n);
                    drain_tokens(s);
                } else {
                    usleep(5000); /* 5ms idle sleep */
                }
            }

            if (mic_interrupted)
                break;

            if (vox_verbose >= 1)
                fprintf(stderr, "Processing...\n");

            vox_stream_finish(s);
            drain_tokens(s);
            fputs("\n", stdout);
            fflush(stdout);

            /* Copy to clipboard if requested */
            if (do_copy && copy_buffer && copy_buffer_len > 0) {
                copy_to_clipboard(copy_buffer);
                if (vox_verbose >= 1)
                    fprintf(stderr, "Copied to clipboard\n");
            }

            /* Reset for next recording */
            free(copy_buffer);
            copy_buffer = NULL;
            copy_buffer_len = 0;

            vox_stream_free(s);
            s = vox_stream_init(ctx);
            if (!s) {
                fprintf(stderr, "Failed to create stream\n");
                break;
            }

            /* Small delay to debounce spacebar release */
            usleep(200000); /* 200ms */
        }

        vox_mic_stop();
        tty_restore();

        /* Skip normal cleanup since we did it in the loop */
        vox_stream_free(s);
        vox_free(ctx);
#ifdef USE_METAL
        vox_metal_shutdown();
#endif
        return 0;
    } else {
        /* File input: load WAV, feed in chunks */
        int n_samples = 0;
        float *samples = vox_load_wav(input_wav, &n_samples);
        if (!samples) {
            fprintf(stderr, "Failed to load %s\n", input_wav);
            vox_stream_free(s);
            vox_free(ctx);
            return 1;
        }
        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_samples, (float)n_samples / VOX_SAMPLE_RATE);

        feed_and_drain(s, samples, n_samples);
        free(samples);
    }

    vox_stream_finish(s);
    drain_tokens(s);
    fputs("\n", stdout);
    fflush(stdout);

    /* Copy to clipboard if requested */
    if (do_copy && copy_buffer && copy_buffer_len > 0) {
        copy_to_clipboard(copy_buffer);
        if (vox_verbose >= 1)
            fprintf(stderr, "Copied to clipboard\n");
    }

    free(copy_buffer);
    vox_stream_free(s);
    vox_free(ctx);
#ifdef USE_METAL
    vox_metal_shutdown();
#endif
    return 0;
}
