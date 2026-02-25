"""Entry point: python -m app

Launches the Unified GUI with calibration and tracking tabs.
"""

from app.config import parse_args, setup_sys_path


def main():
    args = parse_args()
    setup_sys_path()

    from app.core.camera import CameraManager
    from app.core.calibration_store import CalibrationStore
    from app.core.event_loop import MainLoop
    from app.core.demo import OutputModeType
    from app.demos.calibration.demo import CalibrationDemo
    from app.demos.tracking.demo import TrackingDemo
    from app.demos.tracking.gui_output import TrackingGUIOutput

    print("=" * 50)
    print("Unified GUI")
    print("=" * 50)

    # 1. Init hardware (once)
    camera_mgr = CameraManager(args.dvs_camera, args.rgb_camera, args.rgb_rotate)

    print("[INIT] Starting DVS camera...")
    camera_mgr.init_dvs()

    print("[INIT] Starting RGB camera...")
    camera_mgr.init_rgb()

    # 2. Shared calibration store
    cal_store = CalibrationStore()

    # 3. Arm (optional)
    bridge = None
    arm_thread = None
    if not args.no_arm:
        try:
            from app.core.arm import CommandBridge, ArmThread
            bridge = CommandBridge()
            arm_thread = ArmThread(bridge, args.can, args.speed)
            arm_thread.start()
            print("[INIT] Arm thread started")
        except Exception as e:
            print(f"[INIT] Arm init failed: {e} — running without arm")
            bridge = None
            arm_thread = None
    else:
        print("[INIT] Arm disabled (--no-arm)")

    # 4. Create demos
    cal_demo = CalibrationDemo(cal_store, args, bridge=bridge, arm_thread=arm_thread)
    tracking_demo = TrackingDemo(cal_store, args)

    # 5. Register output modes for tracking demo
    tracking_demo.register_output(
        OutputModeType.GUI,
        TrackingGUIOutput(tracking_demo),
    )

    if bridge and arm_thread:
        from app.demos.tracking.phys_dvs_output import TrackingPhysDVSOutput
        from app.demos.tracking.phys_rgb_output import TrackingPhysRGBOutput
        tracking_demo.register_output(
            OutputModeType.PHYS_DVS,
            TrackingPhysDVSOutput(tracking_demo, bridge, arm_thread),
        )
        tracking_demo.register_output(
            OutputModeType.PHYS_RGB,
            TrackingPhysRGBOutput(tracking_demo, bridge, arm_thread),
        )

    # 6. Default output = GUI
    tracking_demo.switch_output(OutputModeType.GUI)

    # 7. Run main loop
    demos = {"Calibration": cal_demo, "Tracking": tracking_demo}
    loop = MainLoop(camera_mgr, demos, bridge=bridge, arm_thread=arm_thread)

    print()
    print("Controls:")
    print("  Click tab bar    — switch tab")
    print("  [q]              — quit")
    print("  [g/e/r]          — GUI / Physical DVS / Physical RGB mode")
    print("  [h]              — arm go home")
    print("  [w]              — arm go center (draw position)")
    print("  [space]          — toggle tracking")
    print("  [c]              — clear canvas")
    print("  [v]              — cycle layout (GUI mode)")
    print("  [Enter]          — confirm calibration")
    print("  [d]              — re-detect RGB quad")
    print()

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")
    finally:
        # Cleanup arm
        if arm_thread:
            arm_thread.stop()
            arm_thread.join(timeout=10.0)
            print("[EXIT] Arm thread stopped")
        print("[EXIT] Done")


if __name__ == "__main__":
    main()
