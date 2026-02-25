from PySide6.QtCore import Qt, QRect, QSize, QPoint
from PySide6.QtWidgets import QLayout, QWidgetItem


class FlowLayout(QLayout):
    """
    Simple wrapping layout (like "flow"/"wrap"): places items left-to-right and
    starts a new row when there is not enough horizontal space.

    IMPORTANT: implements height-for-width so parent layouts can compute the
    correct required height (otherwise lower rows may get clipped).
    """

    def __init__(self, parent=None, margin=0, spacing=8):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    # --- Qt required API ---

    def addItem(self, item):
        self._items.append(item)
        self.invalidate()

    def addWidget(self, w):
        parent = self.parentWidget()
        if parent is not None and w.parent() is None:
            w.setParent(parent)
    
        self.addItem(QWidgetItem(w))

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    # --- layouting ---

    def _do_layout(self, rect, test_only=False):
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())

        x = effective.x()
        y = effective.y()
        line_h = 0
        space = self.spacing()
        max_w = effective.width()

        for item in self._items:
            hint = item.sizeHint()
            w = hint.width()
            h = hint.height()

            # wrap
            if x > effective.x() and (x - effective.x() + w) > max_w:
                x = effective.x()
                y += line_h + space
                line_h = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))

            x += w + space
            line_h = max(line_h, h)

        total_h = (y - effective.y()) + line_h + m.top() + m.bottom()
        return total_h
