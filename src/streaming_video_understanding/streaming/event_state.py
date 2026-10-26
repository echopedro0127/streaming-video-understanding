"""Causal constant-field-count event state, independent of future labels/frames."""
from __future__ import annotations

from dataclasses import dataclass,asdict
import math


@dataclass
class EventCounter:
    enter_frames:int=2
    exit_frames:int=2
    count:int=0
    active:bool=False
    last_timestamp_s:float|None=None
    active_start_s:float|None=None
    active_confirmed_at_s:float|None=None
    enter_streak:int=0
    exit_streak:int=0
    pending_enter_s:float|None=None
    pending_exit_s:float|None=None

    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.enter_frames,self.exit_frames)):
            raise ValueError('confirmation lengths must be positive integers')

    def snapshot(self):
        return asdict(self)

    def update(self,category,timestamp_s):
        if category not in ('presenter','inserted_scene','transition','uncertain',None):
            raise ValueError('unknown category')
        if (isinstance(timestamp_s,bool) or not math.isfinite(timestamp_s) or timestamp_s<0
            or self.last_timestamp_s is not None and timestamp_s<=self.last_timestamp_s):
            raise ValueError('timestamps must be finite, nonnegative and strictly increasing')
        t=float(timestamp_s)
        self.last_timestamp_s=t
        opened,closed=None,None
        if category in ('transition','uncertain',None):
            self.enter_streak=self.exit_streak=0
            self.pending_enter_s=self.pending_exit_s=None
        elif not self.active:
            self.exit_streak=0
            self.pending_exit_s=None
            if category=='presenter':
                self.enter_streak=0
                self.pending_enter_s=None
            else:
                if self.enter_streak==0:
                    self.pending_enter_s=t
                self.enter_streak+=1
                if self.enter_streak>=self.enter_frames:
                    self.active=True
                    self.count+=1
                    self.active_start_s=self.pending_enter_s
                    self.active_confirmed_at_s=t
                    opened=dict(start_s=self.active_start_s,confirmed_at_s=t)
                    self.enter_streak=0
                    self.pending_enter_s=None
        else:
            self.enter_streak=0
            self.pending_enter_s=None
            if category=='inserted_scene':
                self.exit_streak=0
                self.pending_exit_s=None
            else:
                if self.exit_streak==0:
                    self.pending_exit_s=t
                self.exit_streak+=1
                if self.exit_streak>=self.exit_frames:
                    closed=dict(start_s=self.active_start_s,confirmed_at_s=self.active_confirmed_at_s,
                                end_s=self.pending_exit_s,confirmed_end_at_s=t,end_censored=False)
                    self.active=False
                    self.active_start_s=self.active_confirmed_at_s=None
                    self.exit_streak=0
                    self.pending_exit_s=None
        return dict(count=self.count,active=self.active,opened_event=opened,closed_event=closed)
