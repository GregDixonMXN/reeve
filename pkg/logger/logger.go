package logger

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

type Logger struct {
	level  Level
	output *log.Logger
}

func New(level string) *Logger {
	l := &Logger{output: log.New(os.Stdout, "", 0)}
	switch strings.ToLower(level) {
	case "debug":
		l.level = LevelDebug
	case "warn":
		l.level = LevelWarn
	case "error":
		l.level = LevelError
	default:
		l.level = LevelInfo
	}
	return l
}

func (l *Logger) emit(lvl Level, tag, format string, args ...interface{}) {
	if lvl < l.level {
		return
	}
	ts := time.Now().Format("15:04:05.000")
	msg := fmt.Sprintf(format, args...)
	l.output.Printf("%s %s %s", ts, tag, msg)
}

func (l *Logger) Debug(format string, args ...interface{}) {
	l.emit(LevelDebug, "[DBG]", format, args...)
}

func (l *Logger) Info(format string, args ...interface{}) {
	l.emit(LevelInfo, "[INF]", format, args...)
}

func (l *Logger) Warn(format string, args ...interface{}) {
	l.emit(LevelWarn, "[WRN]", format, args...)
}

func (l *Logger) Error(format string, args ...interface{}) {
	l.emit(LevelError, "[ERR]", format, args...)
}
