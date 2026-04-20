
page_css = """
/* 全局容器 */
#home-page {
    display: flex;
    min-height: 100vh;
    background: #f6f7fb;
    position: relative;
    overflow-x: hidden;
}

/* sidebar container including (trigger) */
#sidebar-container {
    position: fixed;
    left: 0;
    top: 0;
    min-width: 100px !important;
    height: 100vh;
    margin-left: 5px;
    width: 80px; /* 初始触发区域宽度，很窄 */
    z-index: 1000;
    transition: width 0.3s;
    box-sizing: border-box;
}

/* 当鼠标靠近左侧或在侧边栏上时，扩大容器宽度以保持显示 */
#sidebar-container:hover {
    width: 200px; 
}

/* 实际的侧边栏 */
#sidebar {
    position: relative;
    margin-left: 5%;
    margin-top: 50px;
    margin-bottom: 50px;
    width: 100%;
    height: 100%;
    padding: 20px 8%;
    box-sizing: border-box;
    
    /* 1. 毛玻璃核心：背景必须是半透明的 */
    background: rgba(214, 245, 247, 0.65) !important; /* 使用 rgba 控制透明度 */
    
    /* 2. 模糊效果：这是毛玻璃的关键 */
    backdrop-filter: blur(15px); 
    -webkit-backdrop-filter: blur(15px); /* 兼容 Safari */

    /* 3. 圆角：因为是左侧边栏，通常只给右侧加圆角 */
    border-radius: 24px 24px 24px 24px;

    /* 4. 边框：加上细微的白色边框，能增加玻璃的质感（高光边） */
    border: 1.5px solid rgba(255, 255, 255, 0.3);
    /* border-left: none; 左侧贴边不需要边框 */

    /* 5. 阴影：让侧边栏有浮起感 */
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.08);
    
    /* 默认隐藏在屏幕左侧外 */
    transform: translateX(-100%);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* 触发弹出 */
#sidebar-container:hover #sidebar {
    transform: translateX(0);
}

/* 内容区域：不再设左边距，自动撑满 */
#content {
    flex: 1;
    width: 100%;
    padding: 40px;
    box-sizing: border-box;
    transition: all 0.3s;
}

/* 按钮样式优化 */
#sidebar .gr-button {
    text-align: left !important;
    justify-content: flex-start !important;
    margin-bottom: 10px;
    border: none !important;
    background: transparent !important;
}

#sidebar .gr-button:hover {
    background: #f0f2f8 !important;
}

"""

# configuration_container_css = """
# #configuration-container {
#     height: 100vh;
#     display: flex;
#     flex-direction: column;
# }
#
# /* scrollable container */
# #scroll-area {
#     flex: 1;
#     overflow-y: auto;
#     padding-bottom: 80px; /*  */
# }
#
# /* unmovable bottom area */
# #footer {
#     position: sticky;
#     bottom: 0;
#     background: white;
#     padding: 10px;
#     border-top: 1px solid #ddd;
# }
# """