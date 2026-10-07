// run-limited: avvia un programma con un tetto rigido alla memoria residente (working set) e priorita' bassa,
// cosi' un modello piu' grande della RAM non sottrae memoria agli altri programmi aperti.
// Le pagine del modello oltre il tetto restano nella cache "standby" di Windows, che cede la precedenza
// alla memoria dei programmi aperti. Durante l'esecuzione misura la RAM disponibile del sistema.
//
// uso: run-limited [--max-ws-mb 3072] [--priority idle|below|normal] [--log-every-s 10] -- programma argomenti...
//   --log-every-s N  ogni N secondi stampa RAM disponibile e working set del processo (0 = mai)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// quoting compatibile con CommandLineToArgvW
static std::wstring quote_arg(const std::wstring & a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) {
        return a;
    }
    std::wstring r = L"\"";
    size_t bs = 0;
    for (wchar_t c : a) {
        if (c == L'\\') {
            bs++;
        } else if (c == L'"') {
            r.append(bs * 2 + 1, L'\\');
            r += L'"';
            bs = 0;
        } else {
            r.append(bs, L'\\');
            r += c;
            bs = 0;
        }
    }
    r.append(bs * 2, L'\\');
    r += L'"';
    return r;
}

static void usage() {
    fprintf(stderr, "uso: run-limited [--max-ws-mb 3072] [--priority idle|below|normal] [--log-every-s 10] -- programma argomenti...\n");
    exit(2);
}

int wmain(int argc, wchar_t ** argv) {
    size_t max_ws_mb = 3072;
    DWORD prio = BELOW_NORMAL_PRIORITY_CLASS;
    unsigned log_every_s = 10;
    int i = 1;
    for (; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (a == L"--") {
            ++i;
            break;
        } else if (a == L"--max-ws-mb" && i + 1 < argc) {
            max_ws_mb = wcstoull(argv[++i], nullptr, 10);
        } else if (a == L"--priority" && i + 1 < argc) {
            const std::wstring p = argv[++i];
            prio = p == L"idle" ? IDLE_PRIORITY_CLASS : p == L"normal" ? NORMAL_PRIORITY_CLASS : BELOW_NORMAL_PRIORITY_CLASS;
        } else if (a == L"--log-every-s" && i + 1 < argc) {
            log_every_s = (unsigned) wcstoul(argv[++i], nullptr, 10);
        } else {
            usage();
        }
    }
    if (i >= argc || max_ws_mb < 64) {
        usage();
    }

    std::wstring cmd;
    for (int k = i; k < argc; ++k) {
        cmd += (k > i ? L" " : L"") + quote_arg(argv[k]);
    }

    // i figli ereditano stdout/stderr anche quando sono pipe o file
    STARTUPINFOW si = {sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    for (HANDLE h : {si.hStdInput, si.hStdOutput, si.hStdError}) {
        if (h && h != INVALID_HANDLE_VALUE) {
            SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        }
    }

    PROCESS_INFORMATION pi = {};
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);
    if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED | prio, nullptr, nullptr, &si, &pi)) {
        fprintf(stderr, "run-limited: CreateProcess fallito (errore %lu)\n", GetLastError());
        return 1;
    }

    const SIZE_T max_ws = (SIZE_T) max_ws_mb << 20;
    if (!SetProcessWorkingSetSizeEx(pi.hProcess, 16u << 20, max_ws, QUOTA_LIMITS_HARDWS_MAX_ENABLE | QUOTA_LIMITS_HARDWS_MIN_DISABLE)) {
        fprintf(stderr, "run-limited: impossibile impostare il tetto di memoria (errore %lu), interrompo\n", GetLastError());
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }

    MEMORYSTATUSEX ms = {sizeof(ms)};
    GlobalMemoryStatusEx(&ms);
    const double avail0 = ms.ullAvailPhys / 1073741824.0;
    double avail_min = avail0;
    fprintf(stderr, "run-limited: tetto working set %zu MB, RAM disponibile all'avvio %.1f GiB\n", max_ws_mb, avail0);

    const ULONGLONG t0 = GetTickCount64();
    ULONGLONG next_log = t0 + log_every_s * 1000ull;
    double avail_min_period = avail0;
    ResumeThread(pi.hThread);
    while (WaitForSingleObject(pi.hProcess, 500) == WAIT_TIMEOUT) {
        GlobalMemoryStatusEx(&ms);
        const double avail = ms.ullAvailPhys / 1073741824.0;
        avail_min = std::min(avail_min, avail);
        avail_min_period = std::min(avail_min_period, avail);
        const ULONGLONG now = GetTickCount64();
        if (log_every_s && now >= next_log) {
            PROCESS_MEMORY_COUNTERS pm = {sizeof(pm)};
            GetProcessMemoryInfo(pi.hProcess, &pm, sizeof(pm));
            fprintf(stderr, "run-limited: t=%5.0f s | RAM disponibile %.1f GiB (minima nel periodo %.1f) | working set %.0f MB\n",
                    (now - t0) / 1000.0, avail, avail_min_period, pm.WorkingSetSize / 1048576.0);
            next_log = now + log_every_s * 1000ull;
            avail_min_period = avail;
        }
    }
    const double dt = (GetTickCount64() - t0) / 1000.0;

    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    PROCESS_MEMORY_COUNTERS pmc = {sizeof(pmc)};
    GetProcessMemoryInfo(pi.hProcess, &pmc, sizeof(pmc));
    fprintf(stderr,
            "run-limited: fine in %.1f s (exit %lu) | picco working set %.0f MB | page fault %lu | "
            "RAM disponibile minima %.1f GiB (all'avvio %.1f)\n",
            dt, code, pmc.PeakWorkingSetSize / 1048576.0, pmc.PageFaultCount, avail_min, avail0);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int) code;
}
