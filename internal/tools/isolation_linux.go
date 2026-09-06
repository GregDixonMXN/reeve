//go:build linux

package tools

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

const (
	landlockCreateRulesetVersion = 1
	landlockRulePathBeneath      = 1
)

type landlockRulesetAttr struct {
	HandledAccessFS  uint64
	HandledAccessNet uint64
}

type landlockPathBeneathAttr struct {
	AllowedAccess uint64
	ParentFD      int32
}

func landlockABI() (int, error) {
	abi, _, errno := unix.Syscall6(
		unix.SYS_LANDLOCK_CREATE_RULESET,
		0,
		0,
		landlockCreateRulesetVersion,
		0,
		0,
		0,
	)
	if errno != 0 {
		return 0, errno
	}
	return int(abi), nil
}

func isolationSupported(allowNetwork bool) error {
	abi, err := landlockABI()
	if err != nil {
		return fmt.Errorf("Landlock unavailable: %w", err)
	}
	if abi < 3 {
		return fmt.Errorf("Landlock ABI %d is too old; ABI 3 or newer is required", abi)
	}
	if !allowNetwork && abi < 4 {
		return fmt.Errorf("Landlock ABI %d cannot enforce TCP network denial; ABI 4 or newer is required", abi)
	}
	return nil
}

func isolationHelperPath() (string, error) {
	const selfExecutable = "/proc/self/exe"
	if _, err := os.Readlink(selfExecutable); err != nil {
		return "", fmt.Errorf("resolve %s: %w", selfExecutable, err)
	}
	return selfExecutable, nil
}

func handledFilesystemAccess(abi int) uint64 {
	access := uint64(
		unix.LANDLOCK_ACCESS_FS_EXECUTE |
			unix.LANDLOCK_ACCESS_FS_WRITE_FILE |
			unix.LANDLOCK_ACCESS_FS_READ_FILE |
			unix.LANDLOCK_ACCESS_FS_READ_DIR |
			unix.LANDLOCK_ACCESS_FS_REMOVE_DIR |
			unix.LANDLOCK_ACCESS_FS_REMOVE_FILE |
			unix.LANDLOCK_ACCESS_FS_MAKE_CHAR |
			unix.LANDLOCK_ACCESS_FS_MAKE_DIR |
			unix.LANDLOCK_ACCESS_FS_MAKE_REG |
			unix.LANDLOCK_ACCESS_FS_MAKE_SOCK |
			unix.LANDLOCK_ACCESS_FS_MAKE_FIFO |
			unix.LANDLOCK_ACCESS_FS_MAKE_BLOCK |
			unix.LANDLOCK_ACCESS_FS_MAKE_SYM,
	)
	if abi >= 2 {
		access |= unix.LANDLOCK_ACCESS_FS_REFER
	}
	if abi >= 3 {
		access |= unix.LANDLOCK_ACCESS_FS_TRUNCATE
	}
	return access
}

func createLandlockRuleset(filesystemAccess, networkAccess uint64) (int, error) {
	attr := landlockRulesetAttr{
		HandledAccessFS:  filesystemAccess,
		HandledAccessNet: networkAccess,
	}
	fd, _, errno := unix.Syscall6(
		unix.SYS_LANDLOCK_CREATE_RULESET,
		uintptr(unsafe.Pointer(&attr)),
		unsafe.Sizeof(attr),
		0,
		0,
		0,
		0,
	)
	if errno != 0 {
		return -1, errno
	}
	return int(fd), nil
}

func addLandlockPathRule(rulesetFD int, path string, access uint64) error {
	fd, err := unix.Open(path, unix.O_PATH|unix.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer unix.Close(fd)

	attr := landlockPathBeneathAttr{AllowedAccess: access, ParentFD: int32(fd)}
	_, _, errno := unix.Syscall6(
		unix.SYS_LANDLOCK_ADD_RULE,
		uintptr(rulesetFD),
		landlockRulePathBeneath,
		uintptr(unsafe.Pointer(&attr)),
		0,
		0,
		0,
	)
	if errno != 0 {
		return errno
	}
	return nil
}

func canonicalExistingPaths(paths []string) []string {
	seen := make(map[string]struct{}, len(paths))
	result := make([]string, 0, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		if strings.HasPrefix(path, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			path = filepath.Join(home, path[2:])
		}
		resolved, err := filepath.EvalSymlinks(path)
		if err != nil {
			continue
		}
		resolved, err = filepath.Abs(resolved)
		if err != nil {
			continue
		}
		if _, ok := seen[resolved]; ok {
			continue
		}
		seen[resolved] = struct{}{}
		result = append(result, resolved)
	}
	return result
}

func readonlyRuntimePaths(commandPath, invocationPath string) []string {
	paths := []string{
		"/usr", "/bin", "/lib", "/lib64",
		"/etc/ld.so.cache", "/etc/ld.so.conf", "/etc/ld.so.conf.d",
		"/etc/nsswitch.conf", "/etc/hosts", "/etc/resolv.conf",
		"/etc/passwd", "/etc/group", "/etc/localtime", "/etc/gitconfig",
		"/etc/ssl", "/etc/ca-certificates",
		commandPath,
	}
	if invocationPath != commandPath {
		paths = append(paths, invocationPath)
	}
	venvRoot := filepath.Dir(filepath.Dir(invocationPath))
	if _, err := os.Stat(filepath.Join(venvRoot, "pyvenv.cfg")); err == nil {
		paths = append(paths, venvRoot)
	}
	// Do not grant the whole PATH, CARGO_HOME, or VIRTUAL_ENV: they may point at
	// workspace-controlled files or unrelated credentials. The selected
	// executable is granted explicitly, and a Python venv is discovered from
	// that invocation path rather than trusted from the environment.
	for _, key := range []string{"GOROOT", "RUSTUP_HOME", "NODE_PATH"} {
		paths = append(paths, os.Getenv(key))
	}
	// Resolve the effective GOROOT from the toolchain itself. `go env`
	// honors GOTOOLCHAIN switches, where $GOROOT is usually unset and the
	// real root may live in the module cache or /opt. Without this, the
	// compile tool is invisible inside the sandbox and isolated builds fail.
	if filepath.Base(commandPath) == "go" {
		if out, err := exec.Command(commandPath, "env", "GOROOT").Output(); err == nil {
			if root := strings.TrimSpace(string(out)); filepath.IsAbs(root) {
				paths = append(paths, root)
			}
		}
	}
	if goPath := os.Getenv("GOPATH"); goPath != "" {
		for _, root := range filepath.SplitList(goPath) {
			paths = append(paths, filepath.Join(root, "bin"), filepath.Join(root, "pkg", "mod"))
		}
	}
	return canonicalExistingPaths(paths)
}

func validateIsolationPaths(request isolationRequest) (isolationRequest, error) {
	allowed := canonicalExistingPaths(request.AllowedDirs)
	if len(allowed) == 0 {
		return isolationRequest{}, fmt.Errorf("no existing allowed directories")
	}
	workingDir, err := filepath.EvalSymlinks(request.WorkingDir)
	if err != nil {
		return isolationRequest{}, fmt.Errorf("resolve working directory: %w", err)
	}
	workingDir, err = filepath.Abs(workingDir)
	if err != nil {
		return isolationRequest{}, fmt.Errorf("canonicalize working directory: %w", err)
	}
	inside := false
	for _, root := range allowed {
		if pathWithin(workingDir, root) {
			inside = true
			break
		}
	}
	if !inside {
		return isolationRequest{}, fmt.Errorf("working directory %q is outside the allowed roots", workingDir)
	}
	request.AllowedDirs = allowed
	request.WorkingDir = workingDir
	return request, nil
}

func applyLandlock(request isolationRequest, commandPath, invocationPath string) error {
	abi, err := landlockABI()
	if err != nil {
		return err
	}
	filesystemAccess := handledFilesystemAccess(abi)
	var networkAccess uint64
	if !request.AllowNetwork {
		networkAccess = unix.LANDLOCK_ACCESS_NET_BIND_TCP | unix.LANDLOCK_ACCESS_NET_CONNECT_TCP
	}
	rulesetFD, err := createLandlockRuleset(filesystemAccess, networkAccess)
	if err != nil {
		return err
	}
	defer unix.Close(rulesetFD)

	readonlyAccess := uint64(unix.LANDLOCK_ACCESS_FS_EXECUTE | unix.LANDLOCK_ACCESS_FS_READ_FILE | unix.LANDLOCK_ACCESS_FS_READ_DIR)
	for _, path := range readonlyRuntimePaths(commandPath, invocationPath) {
		access := readonlyAccess
		if info, err := os.Stat(path); err == nil && !info.IsDir() {
			access = unix.LANDLOCK_ACCESS_FS_EXECUTE | unix.LANDLOCK_ACCESS_FS_READ_FILE
		}
		if err := addLandlockPathRule(rulesetFD, path, access); err != nil {
			return fmt.Errorf("allow runtime path %s: %w", path, err)
		}
	}
	for _, path := range request.AllowedDirs {
		if err := addLandlockPathRule(rulesetFD, path, filesystemAccess); err != nil {
			return fmt.Errorf("allow workspace path %s: %w", path, err)
		}
	}
	deviceAccess := uint64(unix.LANDLOCK_ACCESS_FS_READ_FILE | unix.LANDLOCK_ACCESS_FS_WRITE_FILE)
	for _, path := range []string{"/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"} {
		if err := addLandlockPathRule(rulesetFD, path, deviceAccess); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("allow device %s: %w", path, err)
		}
	}

	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		return fmt.Errorf("set no_new_privs: %w", err)
	}
	_, _, errno := unix.Syscall6(
		unix.SYS_LANDLOCK_RESTRICT_SELF,
		uintptr(rulesetFD),
		0,
		0,
		0,
		0,
		0,
	)
	if errno != 0 {
		return errno
	}
	return nil
}

func nativeAuditArchitecture() (uint32, error) {
	switch runtime.GOARCH {
	case "amd64":
		return unix.AUDIT_ARCH_X86_64, nil
	case "386":
		return unix.AUDIT_ARCH_I386, nil
	case "arm64":
		return unix.AUDIT_ARCH_AARCH64, nil
	case "arm":
		return unix.AUDIT_ARCH_ARM, nil
	case "riscv64":
		return unix.AUDIT_ARCH_RISCV64, nil
	default:
		return 0, fmt.Errorf("seccomp architecture %s is unsupported", runtime.GOARCH)
	}
}

func seccompErrnoFilter(syscalls []uintptr, auditArch uint32) []unix.SockFilter {
	filter := []unix.SockFilter{
		{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 4}, // offsetof(seccomp_data, arch)
		{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jt: 1, K: auditArch},
		{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_KILL_PROCESS},
		{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 0}, // offsetof(seccomp_data, nr)
	}
	if runtime.GOARCH == "amd64" {
		// x32 shares AUDIT_ARCH_X86_64 but adds this bit to syscall numbers.
		filter = append(filter,
			unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JGE | unix.BPF_K, Jf: 1, K: 0x40000000},
			unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_KILL_PROCESS},
		)
	}
	for _, number := range syscalls {
		filter = append(filter,
			unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: uint32(number)},
			unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_ERRNO | uint32(unix.EPERM)},
		)
	}
	return append(filter, unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_ALLOW})
}

func applySeccomp(allowNetwork bool) error {
	auditArch, err := nativeAuditArchitecture()
	if err != nil {
		return err
	}
	blocked := []uintptr{
		unix.SYS_PTRACE,
		unix.SYS_PROCESS_VM_READV,
		unix.SYS_PROCESS_VM_WRITEV,
		unix.SYS_KILL,
		unix.SYS_TKILL,
		unix.SYS_TGKILL,
		unix.SYS_PIDFD_SEND_SIGNAL,
		unix.SYS_BPF,
		unix.SYS_PERF_EVENT_OPEN,
		unix.SYS_USERFAULTFD,
		unix.SYS_IO_URING_SETUP,
		unix.SYS_UNSHARE,
		unix.SYS_SETNS,
		unix.SYS_MOUNT,
		unix.SYS_UMOUNT2,
		unix.SYS_PIVOT_ROOT,
		unix.SYS_CHROOT,
		unix.SYS_KEYCTL,
		unix.SYS_ADD_KEY,
		unix.SYS_REQUEST_KEY,
		// Landlock ABI 4 does not mediate metadata-only mutations. Deny them
		// explicitly so paths outside the allowed roots cannot be chmod/chown,
		// timestamped, or modified through extended attributes.
		unix.SYS_CHMOD,
		unix.SYS_FCHMOD,
		unix.SYS_FCHMODAT,
		unix.SYS_FCHMODAT2,
		unix.SYS_CHOWN,
		unix.SYS_FCHOWN,
		unix.SYS_LCHOWN,
		unix.SYS_FCHOWNAT,
		unix.SYS_UTIME,
		unix.SYS_UTIMES,
		unix.SYS_FUTIMESAT,
		unix.SYS_UTIMENSAT,
		unix.SYS_SETXATTR,
		unix.SYS_LSETXATTR,
		unix.SYS_FSETXATTR,
		unix.SYS_REMOVEXATTR,
		unix.SYS_LREMOVEXATTR,
		unix.SYS_FREMOVEXATTR,
		unix.SYS_SETXATTRAT,
		unix.SYS_REMOVEXATTRAT,
	}
	if runtime.GOARCH == "386" || runtime.GOARCH == "arm" {
		blocked = append(blocked, uintptr(412)) // utimensat_time64
	}
	if !allowNetwork {
		blocked = append(blocked,
			unix.SYS_SOCKET,
			unix.SYS_SOCKETPAIR,
			unix.SYS_CONNECT,
			unix.SYS_BIND,
			unix.SYS_LISTEN,
			unix.SYS_ACCEPT,
			unix.SYS_ACCEPT4,
			unix.SYS_SENDTO,
			unix.SYS_SENDMSG,
			unix.SYS_SENDMMSG,
			unix.SYS_RECVFROM,
			unix.SYS_RECVMSG,
			unix.SYS_RECVMMSG,
		)
	}
	filter := seccompErrnoFilter(blocked, auditArch)
	program := unix.SockFprog{Len: uint16(len(filter)), Filter: &filter[0]}
	if err := unix.Prctl(unix.PR_SET_SECCOMP, unix.SECCOMP_MODE_FILTER, uintptr(unsafe.Pointer(&program)), 0, 0); err != nil {
		return fmt.Errorf("install seccomp filter: %w", err)
	}
	return nil
}

func normalizeResourceLimits(limits isolationResourceLimits) isolationResourceLimits {
	if limits.CPUTimeSec <= 0 {
		limits.CPUTimeSec = defaultSandboxCPUTimeSec
	}
	if limits.MaxMemoryBytes <= 0 {
		limits.MaxMemoryBytes = defaultSandboxMaxMemoryBytes
	}
	if limits.MaxProcesses <= 0 {
		limits.MaxProcesses = defaultSandboxMaxProcesses
	}
	if limits.MaxFileSizeBytes <= 0 {
		limits.MaxFileSizeBytes = defaultSandboxMaxFileSize
	}
	if limits.MaxOpenFiles <= 0 {
		limits.MaxOpenFiles = defaultSandboxMaxOpenFiles
	}
	return limits
}

func setResourceLimit(resource int, requested uint64) error {
	var inherited unix.Rlimit
	if err := unix.Getrlimit(resource, &inherited); err != nil {
		return err
	}
	value := requested
	if inherited.Max != unix.RLIM_INFINITY && value > inherited.Max {
		value = inherited.Max
	}
	return unix.Setrlimit(resource, &unix.Rlimit{Cur: value, Max: value})
}

func applyResourceLimits(limits isolationResourceLimits) error {
	limits = normalizeResourceLimits(limits)
	settings := []struct {
		name     string
		resource int
		value    uint64
	}{
		{name: "CPU time", resource: unix.RLIMIT_CPU, value: uint64(limits.CPUTimeSec)},
		{name: "address space", resource: unix.RLIMIT_AS, value: uint64(limits.MaxMemoryBytes)},
		{name: "process count", resource: unix.RLIMIT_NPROC, value: uint64(limits.MaxProcesses)},
		{name: "file size", resource: unix.RLIMIT_FSIZE, value: uint64(limits.MaxFileSizeBytes)},
		{name: "open files", resource: unix.RLIMIT_NOFILE, value: uint64(limits.MaxOpenFiles)},
		{name: "core dump", resource: unix.RLIMIT_CORE, value: 0},
	}
	for _, setting := range settings {
		if err := setResourceLimit(setting.resource, setting.value); err != nil {
			return fmt.Errorf("set %s resource limit: %w", setting.name, err)
		}
	}
	return nil
}

func runIsolated(request isolationRequest) error {
	if err := isolationSupported(request.AllowNetwork); err != nil {
		return err
	}
	request, err := validateIsolationPaths(request)
	if err != nil {
		return err
	}
	if err := os.Chdir(request.WorkingDir); err != nil {
		return fmt.Errorf("change working directory: %w", err)
	}
	invocationPath, err := exec.LookPath(request.Command[0])
	if err != nil {
		return fmt.Errorf("resolve command: %w", err)
	}
	invocationPath, err = filepath.Abs(invocationPath)
	if err != nil {
		return fmt.Errorf("canonicalize command invocation: %w", err)
	}
	commandPath, err := filepath.EvalSymlinks(invocationPath)
	if err != nil {
		return fmt.Errorf("resolve command symlinks: %w", err)
	}
	request.Command[0] = invocationPath
	if err := applyResourceLimits(request.ResourceLimits); err != nil {
		return err
	}
	if err := applyLandlock(request, commandPath, invocationPath); err != nil {
		return fmt.Errorf("apply Landlock: %w", err)
	}
	if err := applySeccomp(request.AllowNetwork); err != nil {
		return err
	}
	return syscall.Exec(invocationPath, request.Command, os.Environ())
}
