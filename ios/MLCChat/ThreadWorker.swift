//
//  ThreadWorker.swift
//  MLCChat

import Foundation

// A simple thread worker that is backed by a single thread
//
// Instead of dispatch queue, we need a dedicated thread for metal compute
// so all thread local resources are centralized at a single thread
class ThreadWorker : Thread {
    private var cond = NSCondition();
    private var queue = Array<()->Void>();
    
    override func main()  {
        Thread.setThreadPriority(1)
        while (true) {
            self.cond.lock()
            while (queue.isEmpty) {
                self.cond.wait()
            }
            let task = self.queue.removeFirst()
            self.cond.unlock()
            task()
        }
    }
    
    func push(task: @escaping ()->Void) {
        self.cond.lock()
        self.queue.append(task)
        self.cond.signal()
        self.cond.unlock()

    }
}
