//go:build !linux

package tools

import "fmt"

func isolationSupported(bool) error {
	return fmt.Errorf("OS process isolation is currently supported only on Linux")
}

func isolationHelperPath() (string, error) {
	return "", isolationSupported(false)
}

func runIsolated(isolationRequest) error {
	return isolationSupported(false)
}
