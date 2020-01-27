package tree;

import java.util.ArrayList;
import java.util.List;

public class TreeNode {

	private String data = null;

	private List<TreeNode> children = new ArrayList<>();

	private TreeNode parent = null;

	public TreeNode(String data) {
		this.data = data;
	}

	public void addChildSorted(TreeNode child) {
		boolean inserted = false;
		int i = 0;
		while (i<children.size() && !inserted) {
			if (children.get(i).getData().compareTo(child.getData())<=0) {
				children.add(i, child);
				inserted=true;
			}
			i++;
		}
		if (!inserted) {
			children.add(child);
		}
	}
	
	public void addChild(TreeNode child){
		child.setParent(this);
		this.children.add(child);
	}

	public void addChildren(List<TreeNode> children) {
		children.forEach(each -> each.setParent(this));
		this.children.addAll(children);
	}

	public List<TreeNode> getChildren() {
		return children;
	}

	public String getData() {
		return data;
	}

	public void setData(String data) {
		this.data = data;
	}

	private void setParent(TreeNode parent) {
		this.parent = parent;
	}

	public TreeNode getParent() {
		return parent;
	}

	public void deleteNode() {
		if (parent != null) {
			int index = this.parent.getChildren().indexOf(this);
			this.parent.getChildren().remove(this);
			for (TreeNode each : getChildren()) {
				each.setParent(this.parent);
			}
			this.parent.getChildren().addAll(index, this.getChildren());
		} else {
			deleteRootNode();
		}
		this.getChildren().clear();
	}

	public TreeNode deleteRootNode() {
		if (parent != null) {
			throw new IllegalStateException("deleteRootNode not called on root");
		}
		TreeNode newParent = null;
		if (!getChildren().isEmpty()) {
			newParent = getChildren().get(0);
			newParent.setParent(null);
			getChildren().remove(0);
			for (TreeNode each : getChildren()) {
				each.setParent(newParent);
			}
			newParent.getChildren().addAll(getChildren());
		}
		this.getChildren().clear();
		return newParent;
	}

	public TreeNode getRoot() {
		if (parent == null) {
			return this;
		}
		return parent.getRoot();
	}
	
	public int size() {
		if (this.children!=null && this.children.size()==0) {
			return 1;
		} else {
			return 1 + size(children);
		}
	}

	private int size(List<TreeNode> children) {
		int size = 0;
		for (TreeNode subtree : children) {
			size += subtree.size();
		}
		return size;
	}

}